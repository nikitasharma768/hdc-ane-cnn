// ContentView.swift
// CNN benchmark app for iPad / iPhone (works in Swift Playgrounds or Xcode).
//
// Same method as bench.swift and Sri's ContentView.swift: fixed-size batches,
// timed runs over the whole test set (fill input + predict + argmax), then
// average time per run and accuracy. Runs every model in the chosen folder
// on CPU, GPU, and Neural Engine, one after another.
//
// "Power hold" keeps each test running for N seconds (with 2 s idle before
// and after) so Power Profiler has a long, clean stretch to measure. The
// hold start/end times are saved in the CSV so you can find each test in
// the Power Profiler timeline.
//
// The chosen folder must contain the CNN_Batch*_Inference.mlpackage models
// and digits_test.data.

import SwiftUI
import CoreML
import UIKit
import UniformTypeIdentifiers

// MARK: - Result row

struct ResultRow: Identifiable, Sendable {
    let id = UUID()
    let unit: String
    let batch: Int
    let avgMs: Double
    let accuracy: Double
    let timedRuns: Int
    let holdStart: Date?
    let holdEnd: Date?
    let holdRuns: Int
}

// MARK: - Data loading (same format as Sri's loadISOLET)

func loadData(url: URL) -> ([[Double]], [Int]) {
    guard let content = try? String(contentsOf: url, encoding: .utf8) else { return ([], []) }
    var features: [[Double]] = []
    var labels: [Int] = []
    for line in content.components(separatedBy: .newlines) {
        if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
        let values = line.split(separator: ",").compactMap {
            Double($0.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        if values.count < 2 { continue }
        features.append(Array(values.dropLast()))
        labels.append(Int(values.last!) - 1)
    }
    return (features, labels)
}

func batchSize(from url: URL) -> Int? {
    let name = url.deletingPathExtension().lastPathComponent
    guard let r = name.range(of: "Batch") else { return nil }
    return Int(name[r.upperBound...].prefix(while: { $0.isNumber }))
}

func computeUnits(_ unit: String) -> MLComputeUnits {
    switch unit {
    case "cpu": return .cpuOnly
    case "gpu": return .cpuAndGPU
    default: return .cpuAndNeuralEngine
    }
}

// One pass over the whole dataset. Returns the number of correct predictions.
func runDataset(model: MLModel, inputName: String, outputName: String,
                features: [[Double]], labels: [Int], batchSize: Int) throws -> Int {
    let numSamples = features.count
    let inputLength = features[0].count
    var correct = 0
    var batchStart = 0

    while batchStart < numSamples {
        let currentBatch = min(batchSize, numSamples - batchStart)
        let inputArray = try MLMultiArray(
            shape: [NSNumber(value: batchSize), NSNumber(value: inputLength)],
            dataType: .double
        )
        for i in 0..<currentBatch {
            for j in 0..<inputLength {
                inputArray[[NSNumber(value: i), NSNumber(value: j)]] =
                    NSNumber(value: features[batchStart + i][j])
            }
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLFeatureValue(multiArray: inputArray)]
        )
        let prediction = try model.prediction(from: provider)
        guard let output = prediction.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "Bench", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Missing output \(outputName)"])
        }
        let nClasses = output.shape[1].intValue
        for i in 0..<currentBatch {
            var maxVal = -Double.greatestFiniteMagnitude
            var predicted = 0
            for j in 0..<nClasses {
                let val = output[[NSNumber(value: i), NSNumber(value: j)]].doubleValue
                if val > maxVal {
                    maxVal = val
                    predicted = j
                }
            }
            if predicted == labels[batchStart + i] { correct += 1 }
        }
        batchStart += currentBatch
    }
    return correct
}

func elapsedMs(since start: DispatchTime) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
}

// MARK: - Benchmark runner

func runBenchmarks(folder: URL, units: [String], runs: Int, holdSeconds: Double,
                   status: @escaping @Sendable (String) async -> Void,
                   result: @escaping @Sendable (ResultRow) async -> Void) async {
    let access = folder.startAccessingSecurityScopedResource()
    defer { if access { folder.stopAccessingSecurityScopedResource() } }

    let (features, labels) = loadData(url: folder.appendingPathComponent("digits_test.data"))
    guard !features.isEmpty else {
        await status("Could not read digits_test.data in the chosen folder.")
        return
    }

    guard let items = try? FileManager.default.contentsOfDirectory(
        at: folder, includingPropertiesForKeys: nil) else {
        await status("Could not open the chosen folder.")
        return
    }
    let models = items
        .filter { $0.pathExtension == "mlpackage" }
        .compactMap { url -> (Int, URL)? in
            guard let b = batchSize(from: url) else { return nil }
            return (b, url)
        }
        .sorted { $0.0 < $1.0 }
    guard !models.isEmpty else {
        await status("No .mlpackage models found in the chosen folder.")
        return
    }

    // Compile every model once
    var compiled: [(Int, URL)] = []
    for (b, url) in models {
        await status("Compiling batch \(b)…")
        do {
            let compiledURL = try await MLModel.compileModel(at: url)
            compiled.append((b, compiledURL))
        } catch {
            await status("Compile failed for batch \(b): \(error.localizedDescription)")
            return
        }
    }

    var finished = 0
    for unit in units {
        for (b, compiledURL) in compiled {
            let label = "\(unit.uppercased()) batch \(b)"
            do {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits(unit)
                let model = try MLModel(contentsOf: compiledURL, configuration: config)
                guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
                      let outputName = model.modelDescription.outputDescriptionsByName.keys.first else {
                    await status("\(label): could not read model input/output names")
                    continue
                }

                // Warm-up run (not timed)
                await status("\(label): warming up…")
                _ = try runDataset(model: model, inputName: inputName, outputName: outputName,
                                   features: features, labels: labels, batchSize: b)

                // Timed runs
                await status("\(label): \(runs) timed runs…")
                var times: [Double] = []
                var correct = 0
                for _ in 0..<runs {
                    let t0 = DispatchTime.now()
                    correct += try runDataset(model: model, inputName: inputName, outputName: outputName,
                                              features: features, labels: labels, batchSize: b)
                    times.append(elapsedMs(since: t0))
                }
                let avg = times.reduce(0, +) / Double(runs)
                let acc = Double(correct) / Double(features.count * runs) * 100

                // Power hold
                var holdStart: Date? = nil
                var holdEnd: Date? = nil
                var holdRuns = 0
                if holdSeconds > 0 {
                    await status("\(label): power hold \(Int(holdSeconds)) s…")
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                    let start = Date()
                    while Date().timeIntervalSince(start) < holdSeconds {
                        _ = try runDataset(model: model, inputName: inputName, outputName: outputName,
                                           features: features, labels: labels, batchSize: b)
                        holdRuns += 1
                    }
                    holdStart = start
                    holdEnd = Date()
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                }

                await result(ResultRow(unit: unit, batch: b, avgMs: avg, accuracy: acc,
                                       timedRuns: runs, holdStart: holdStart, holdEnd: holdEnd,
                                       holdRuns: holdRuns))
                finished += 1
            } catch {
                await status("\(label) failed: \(error.localizedDescription)")
            }
        }
    }
    await status("Done. \(finished) tests finished.")
}

// MARK: - View model

@MainActor
final class BenchViewModel: ObservableObject {
    @Published var folderURL: URL?
    @Published var runs = 20
    @Published var holdSeconds: Double = 10
    @Published var useCPU = true
    @Published var useGPU = true
    @Published var useANE = true
    @Published var isRunning = false
    @Published var status = "Choose the CNNModels folder to begin."
    @Published var results: [ResultRow] = []

    var csv: String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        f.timeZone = .current
        let device = "\(UIDevice.current.model) \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
        var text = "# device: \(device)\n"
        text += "unit,batch,avg_ms,accuracy_pct,timed_runs,hold_start,hold_end,hold_runs\n"
        for r in results {
            let hs = r.holdStart.map { f.string(from: $0) } ?? ""
            let he = r.holdEnd.map { f.string(from: $0) } ?? ""
            text += "\(r.unit),\(r.batch),\(String(format: "%.3f", r.avgMs)),"
            text += "\(String(format: "%.2f", r.accuracy)),\(r.timedRuns),\(hs),\(he),\(r.holdRuns)\n"
        }
        return text
    }

    func start() {
        guard let folder = folderURL else { return }
        var units: [String] = []
        if useCPU { units.append("cpu") }
        if useGPU { units.append("gpu") }
        if useANE { units.append("ane") }
        guard !units.isEmpty else { return }

        let runs = self.runs
        let hold = self.holdSeconds
        let vm = self
        isRunning = true
        results = []
        UIApplication.shared.isIdleTimerDisabled = true   // keep the screen on

        Task.detached(priority: .userInitiated) {
            await runBenchmarks(
                folder: folder, units: units, runs: runs, holdSeconds: hold,
                status: { msg in await MainActor.run { vm.status = msg } },
                result: { row in await MainActor.run { vm.results.append(row) } }
            )
            await MainActor.run {
                vm.isRunning = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

// MARK: - Screen

struct ContentView: View {
    @StateObject private var vm = BenchViewModel()
    @State private var showPicker = false

    var body: some View {
        NavigationStack {
            Form {
                Section("1. Models folder") {
                    Button(vm.folderURL?.lastPathComponent ?? "Choose CNNModels folder") {
                        showPicker = true
                    }
                    .disabled(vm.isRunning)
                }

                Section("2. Settings") {
                    Stepper("Timed runs: \(vm.runs)", value: $vm.runs, in: 5...100, step: 5)
                    Stepper("Power hold: \(Int(vm.holdSeconds)) s", value: $vm.holdSeconds, in: 0...30, step: 5)
                    Toggle("CPU", isOn: $vm.useCPU)
                    Toggle("GPU", isOn: $vm.useGPU)
                    Toggle("Neural Engine", isOn: $vm.useANE)
                }
                .disabled(vm.isRunning)

                Section("3. Run") {
                    Button(vm.isRunning ? "Running…" : "Run all tests") { vm.start() }
                        .disabled(vm.isRunning || vm.folderURL == nil)
                    Text(vm.status)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                if !vm.results.isEmpty {
                    Section("Results") {
                        ForEach(vm.results) { r in
                            HStack {
                                Text("\(r.unit.uppercased())  batch \(r.batch)")
                                Spacer()
                                Text(String(format: "%.3f ms   %.2f%%", r.avgMs, r.accuracy))
                                    .monospacedDigit()
                            }
                        }
                        ShareLink(item: vm.csv) {
                            Label("Share results (CSV)", systemImage: "square.and.arrow.up")
                        }
                    }
                }
            }
            .navigationTitle("CNN Benchmark")
        }
        .fileImporter(isPresented: $showPicker, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result {
                vm.folderURL = url
                vm.status = "Folder chosen. Tap Run all tests."
            }
        }
    }
}
