// bench.swift
// Command-line version of Sri's ContentView.swift benchmark.
// Same method: fixed-size batches, 20 runs over the whole dataset, each run
// timed end to end (fill input + predict + argmax), then average + accuracy.
// Differences: runs from Terminal (no Xcode needed), and the model, dataset,
// batch size, and compute unit are passed as arguments instead of hard-coded.
//
// Build:  swiftc -O bench.swift -o bench
// Run:    ./bench <model.mlpackage> <data file> <batch size> <cpu|gpu|ane|all> [runs]
// Example ./bench CNNModels/CNN_Batch256_Inference.mlpackage digits_test.data 256 ane

import Foundation
import CoreML

struct BenchError: Error, CustomStringConvertible {
    let description: String
}

// Same parser as Sri's loadISOLET: comma-separated features, label last (1-indexed)
func loadData(path: String) -> ([[Double]], [Int]) {
    do {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        var features: [[Double]] = []
        var labels: [Int] = []

        for (lineIndex, line) in content.components(separatedBy: .newlines).enumerated() {
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let parts = line.split(separator: ",")
            if parts.count < 2 {
                print("Bad row at line \(lineIndex)")
                continue
            }
            var rowValues: [Double] = []
            for part in parts {
                let cleaned = part.trimmingCharacters(in: .whitespacesAndNewlines)
                guard let val = Double(cleaned) else {
                    print("FAILED TO PARSE: \(cleaned) (line \(lineIndex))")
                    rowValues.removeAll()
                    break
                }
                rowValues.append(val)
            }
            if rowValues.isEmpty { continue }
            features.append(Array(rowValues.dropLast()))
            labels.append(Int(rowValues.last!) - 1)
        }
        return (features, labels)
    } catch {
        print("LOAD ERROR: \(error)")
        return ([], [])
    }
}

// MARK: - Arguments

let args = CommandLine.arguments
if args.count < 5 {
    print("Usage: ./bench <model.mlpackage> <data file> <batch size> <cpu|gpu|ane|all> [runs]")
    exit(1)
}
let modelPath = args[1]
let dataPath = args[2]
guard let batchSize = Int(args[3]), batchSize > 0 else {
    print("Batch size must be a positive number")
    exit(1)
}
let unitsArg = args[4].lowercased()
let numRuns = args.count > 5 ? (Int(args[5]) ?? 20) : 20

let config = MLModelConfiguration()
switch unitsArg {
case "cpu": config.computeUnits = .cpuOnly
case "gpu": config.computeUnits = .cpuAndGPU
case "ane": config.computeUnits = .cpuAndNeuralEngine
case "all": config.computeUnits = .all
default:
    print("Compute unit must be cpu, gpu, ane, or all")
    exit(1)
}

// MARK: - Benchmark

do {
    let (features, labels) = loadData(path: dataPath)
    guard !features.isEmpty else { throw BenchError(description: "No data loaded from \(dataPath)") }
    let numSamples = features.count
    let inputLength = features[0].count

    let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
    let model = try MLModel(contentsOf: compiledURL, configuration: config)

    guard let inputName = model.modelDescription.inputDescriptionsByName.keys.first,
          let outputName = model.modelDescription.outputDescriptionsByName.keys.first else {
        throw BenchError(description: "Could not read model input/output names")
    }

    print("Model: \(modelPath)")
    print("Data: \(dataPath)  Samples: \(numSamples)  Features: \(inputLength)")
    print("Batch size: \(batchSize)  Compute: \(unitsArg)  Runs: \(numRuns)")
    print("Input: \(inputName)  Output: \(outputName)")
    print("Start time: \(Date())")

    var timesPerRun: [Double] = []
    var totalCorrect = 0

    for run in 1...numRuns {
        var correctThisRun = 0
        let runStart = DispatchTime.now()
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
                throw BenchError(description: "Missing output \(outputName)")
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
                if predicted == labels[batchStart + i] { correctThisRun += 1 }
            }
            batchStart += currentBatch
        }

        let runMs = Double(DispatchTime.now().uptimeNanoseconds - runStart.uptimeNanoseconds) / 1_000_000
        timesPerRun.append(runMs)
        totalCorrect += correctThisRun
        print("Run \(run): \(String(format: "%.3f", runMs)) ms")
    }

    let averageMs = timesPerRun.reduce(0, +) / Double(numRuns)
    let accuracy = Double(totalCorrect) / Double(numSamples * numRuns) * 100

    print("End time: \(Date())")
    print("Average Runtime: \(String(format: "%.3f", averageMs)) ms")
    print("Accuracy: \(String(format: "%.2f", accuracy))%")
} catch {
    print("Error: \(error)")
    exit(1)
}
