import SwiftUI
import CoreML

// MARK: - CSV Loader
func loadCSV(path: String) -> [[Double]] {
    do {
        let content = try String(contentsOfFile: path, encoding: .utf8)
        let rows = content.split(separator: "\n")
        return rows.map { row in
            row.split(separator: ",").compactMap { Double($0) }
        }
    } catch {
        print("Error loading CSV: \(error)")
        return []
    }
}

struct ContentView: View {
    
    // MARK: - Compute Prediction Once
    let predictionText: String = {
        do {
            // Load CSV files from bundle
            guard let imagesPath = Bundle.main.path(forResource: "digits_images", ofType: "csv"),
                  let labelsPath = Bundle.main.path(forResource: "digits_labels", ofType: "csv") else {
                return "Error: CSV files not found in app bundle."
            }
            
            let digitsImages = loadCSV(path: imagesPath)
            let digitsLabelsRaw = loadCSV(path: labelsPath)
            let digitsLabels = digitsLabelsRaw.map { Int($0[0]) }
            
            guard !digitsImages.isEmpty && digitsImages.count == digitsLabels.count else {
                return "Error: Dataset not loaded properly."
            }
            
            let numSamples = digitsImages.count
            let inputVectorLength = digitsImages[0].count
            let batchSize = 256
            let numRuns = 20
            
            // Load Core ML model
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            let model = try BatchedInference(configuration: config)
            
            var totalTimesPerRun: [Double] = []
            var totalCorrectPredictionsAcrossRuns = 0
            
            for run in 1...numRuns {
                var correctPredictionsForRun = 0
                let runStart = DispatchTime.now()
                
                var batchStart = 0
                while batchStart < numSamples {
                    let remainingSamples = numSamples - batchStart
                    let currentBatchSize = min(batchSize, remainingSamples)
                    
                    // Create MLMultiArray of fixed batch size [256, 64]
                    let inputArray = try MLMultiArray(shape: [NSNumber(value: batchSize), NSNumber(value: inputVectorLength)], dataType: .double)
                    
                    // Copy real samples into first currentBatchSize rows
                    for i in 0..<currentBatchSize {
                        for j in 0..<inputVectorLength {
                            inputArray[[NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: digitsImages[batchStart + i][j])
                        }
                    }
                    
                    let input = BatchedInferenceInput(X: inputArray)
                    let prediction = try model.prediction(input: input)
                    
                    // Process output (assuming output property is var_18)
                    let outputArray = prediction.var_18
                    let nClasses = outputArray.shape[1].intValue
                    
                    for i in 0..<currentBatchSize {
                        var maxVal: Double = -Double.greatestFiniteMagnitude
                        var predictedClass = 0
                        for j in 0..<nClasses {
                            let val = outputArray[[NSNumber(value: i), NSNumber(value: j)]].doubleValue
                            if val > maxVal {
                                maxVal = val
                                predictedClass = j
                            }
                        }
                        if predictedClass == digitsLabels[batchStart + i] {
                            correctPredictionsForRun += 1
                        }
                    }
                    
                    batchStart += currentBatchSize
                } // end batch loop
                
                let runEnd = DispatchTime.now()
                let runNanoTime = runEnd.uptimeNanoseconds - runStart.uptimeNanoseconds
                let runTimeMs = Double(runNanoTime) / 1_000_000
                totalTimesPerRun.append(runTimeMs)
                totalCorrectPredictionsAcrossRuns += correctPredictionsForRun
            } // end run loop
            
            let averageTotalTimeMs = totalTimesPerRun.reduce(0, +) / Double(numRuns)
            let overallAccuracy = Double(totalCorrectPredictionsAcrossRuns) / Double(numSamples * numRuns) * 100
            
            var perRunText = ""
            for (index, timeMs) in totalTimesPerRun.enumerated() {
                perRunText += "Run \(index+1): \(String(format: "%.3f", timeMs)) ms total\n"
            }
            
            return """
            Processed \(numSamples) samples × \(numRuns) runs = \(numSamples * numRuns) samples.
            
            Total inference time per run:
            \(perRunText)
            
            Average total inference time per run: \(String(format: "%.3f", averageTotalTimeMs)) ms
            Overall Accuracy: \(String(format: "%.2f", overallAccuracy))%
            """
            
        } catch {
            return "Error: \(error)"
        }
    }()
    
    // MARK: - View
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "brain.head.profile")
                .imageScale(.large)
                .foregroundStyle(.tint)
            
            ScrollView {
                Text(predictionText)
                    .font(.headline)
                    .padding()
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
