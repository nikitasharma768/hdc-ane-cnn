import SwiftUI
import CoreML

func loadISOLET(path: String) -> ([[Double]], [Int]) {

    do {

        let content = try String(contentsOfFile: path)

        let lines = content.components(separatedBy: .newlines)

        var features: [[Double]] = []
        var labels: [Int] = []

        for (lineIndex, line) in lines.enumerated() {

            if line.trimmingCharacters(in: .whitespaces).isEmpty {
                continue
            }

            let parts = line.split(separator: ",")

            if parts.count < 2 {
                print("Bad row at line \(lineIndex)")
                continue
            }

            var rowValues: [Double] = []

            for part in parts {

                let cleaned =
                    part
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                guard let val = Double(cleaned) else {

                    print("FAILED TO PARSE:")
                    print(cleaned)
                    print("Line:", lineIndex)

                    rowValues.removeAll()
                    break
                }

                rowValues.append(val)
            }

            if rowValues.isEmpty {
                continue
            }

            let x = Array(rowValues.dropLast())

            let y = Int(rowValues.last!) - 1

            features.append(x)
            labels.append(y)
        }

        print("Loaded rows:", features.count)

        return (features, labels)

    } catch {

        print("LOAD ERROR:", error)

        return ([], [])
    }
}
struct ContentView: View {

    let predictionText: String = {

        do {

            // MARK: - Load ISOLET test dataset

            guard let isoletPath = Bundle.main.path(
                forResource: "isolet5",
                ofType: "data"
            ) else {
                return "Error: isolet5.data not found."
            }

            let (isoletFeatures, isoletLabels) = loadISOLET(path: isoletPath)

            guard !isoletFeatures.isEmpty else {
                return "Error: Failed to load ISOLET."
            }

            let numSamples = isoletFeatures.count
            let inputVectorLength = isoletFeatures[0].count   // should be 617

            let batchSize = 2048
            let numRuns = 20

            // MARK: - Load CoreML Model

            let config = MLModelConfiguration()
            config.computeUnits = .cpuOnly
            
            
            // model,
            let model = try BatchSize2048D2000(configuration: config)

            var totalTimesPerRun: [Double] = []
            var totalCorrectPredictionsAcrossRuns = 0

            // MARK: - Benchmark Loop

            for run in 1...numRuns {

                var correctPredictionsForRun = 0

                let runStart = DispatchTime.now()

                var batchStart = 0

                while batchStart < numSamples {

                    let remainingSamples = numSamples - batchStart
                    let currentBatchSize = min(batchSize, remainingSamples)

                    // Create fixed-size MLMultiArray
                    let inputArray = try MLMultiArray(
                        shape: [
                            NSNumber(value: batchSize),
                            NSNumber(value: inputVectorLength)
                        ],
                        dataType: .double
                    )

                    // Fill valid rows
                    for i in 0..<currentBatchSize {
                        for j in 0..<inputVectorLength {

                            inputArray[
                                [
                                    NSNumber(value: i),
                                    NSNumber(value: j)
                                ]
                            ] = NSNumber(
                                value: isoletFeatures[batchStart + i][j]
                            )
                        }
                    }

                    // MARK: - Run inference

                    let prediction = try model.prediction(X: inputArray)

                    let outputArray = prediction.var_18

                    let nClasses = outputArray.shape[1].intValue

                    // MARK: - Argmax

                    for i in 0..<currentBatchSize {

                        var maxVal = -Double.greatestFiniteMagnitude
                        var predictedClass = 0

                        for j in 0..<nClasses {

                            let val = outputArray[
                                [
                                    NSNumber(value: i),
                                    NSNumber(value: j)
                                ]
                            ].doubleValue

                            if val > maxVal {
                                maxVal = val
                                predictedClass = j
                            }
                        }

                        if predictedClass == isoletLabels[batchStart + i] {
                            correctPredictionsForRun += 1
                        }
                    }

                    batchStart += currentBatchSize
                }

                let runEnd = DispatchTime.now()

                let runNanoTime =
                    runEnd.uptimeNanoseconds - runStart.uptimeNanoseconds

                let runTimeMs = Double(runNanoTime) / 1_000_000

                totalTimesPerRun.append(runTimeMs)

                totalCorrectPredictionsAcrossRuns += correctPredictionsForRun
            }

            // MARK: - Metrics

            let averageTotalTimeMs =
                totalTimesPerRun.reduce(0, +) / Double(numRuns)

            let overallAccuracy =
                Double(totalCorrectPredictionsAcrossRuns)
                / Double(numSamples * numRuns)
                * 100

            var perRunText = ""

            for (index, timeMs) in totalTimesPerRun.enumerated() {

                perRunText +=
                    "Run \(index + 1): \(String(format: "%.3f", timeMs)) ms\n"
            }

            return """
            ISOLET Benchmark

            Samples: \(numSamples)
            Runs: \(numRuns)

            \(perRunText)

            Average Runtime:
            \(String(format: "%.3f", averageTotalTimeMs)) ms

            Accuracy:
            \(String(format: "%.2f", overallAccuracy))%
            """

        } catch {
            return "Error: \(error)"
        }

    }()

    var body: some View {
        ScrollView {
            Text(predictionText)
                .padding()
        }
    }
}
