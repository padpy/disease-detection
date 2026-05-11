import UIKit
import Flutter
import GoogleMaps
import AVFoundation
import CoreImage
import Vision

@main
@objc class AppDelegate: FlutterAppDelegate {
  // Retained so the channel handlers don't get torn down.
  private var vpnController: VpnController?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let controller : FlutterViewController = window?.rootViewController as! FlutterViewController

    // VPN MethodChannel + EventChannel (see VpnController.swift).
    self.vpnController = VpnController(messenger: controller.binaryMessenger)

    let cameraChannel = FlutterMethodChannel(name: "gopher_eye/camera_control",
                                              binaryMessenger: controller.binaryMessenger)
    cameraChannel.setMethodCallHandler({
      (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in
      if call.method == "setLensPosition" {
        if let args = call.arguments as? [String: Any],
           let position = args["position"] as? Double {
             self.setLensPosition(Float(position))
             result(nil)
        } else {
          result(FlutterError(code: "INVALID_ARGUMENT", message: "Position is required", details: nil))
        }
      } else if call.method == "getLensPosition" {
        self.getLensPosition(result: result)
      } else if call.method == "generateFocusStack" {
         if let args = call.arguments as? [String: Any],
            let files = args["files"] as? [String] {
             FocusStacker.process(imagePaths: files) { path, error in
                 if let path = path {
                     result(path)
                 } else {
                     result(FlutterError(code: "STACK_FAILED", message: error?.localizedDescription, details: nil))
                 }
             }
         } else {
             result(FlutterError(code: "INVALID_ARGUMENT", message: "File paths required", details: nil))
         }
      } else {
        result(FlutterMethodNotImplemented)
      }
    })

    let keyPath = Bundle.main.path(forResource: "google_maps_api_key", ofType: "txt")
    let googleMapKey: String
    do {
       googleMapKey = try String(contentsOfFile: keyPath!, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
    } catch {
      print("Error reading Google Maps API key: \(error)")
      return false
    }
      
    GMSServices.provideAPIKey(googleMapKey)// specify your API key in the application delegate
    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  private func setLensPosition(_ position: Float) {
      // Find the active capture device (usually the back camera)
      // Note: This assumes the standard back camera is being used.
      // If the app supports multiple cameras/lens switching, we might need to find the specific one.
      let discoverySession = AVCaptureDevice.DiscoverySession(
          deviceTypes: [.builtInWideAngleCamera],
          mediaType: .video,
          position: .back
      )
      
      if let device = discoverySession.devices.first {
          do {
              try device.lockForConfiguration()
              device.setFocusModeLocked(lensPosition: position, completionHandler: nil)
              device.unlockForConfiguration()
          } catch {
              print("Error locking configuration for lens position: \(error)")
          }
      }
  }

  private func getLensPosition(result: @escaping FlutterResult) {
      let discoverySession = AVCaptureDevice.DiscoverySession(
          deviceTypes: [.builtInWideAngleCamera],
          mediaType: .video,
          position: .back
      )
      
      if let device = discoverySession.devices.first {
          result(device.lensPosition)
      } else {
          result(FlutterError(code: "NO_DEVICE", message: "No camera found", details: nil))
      }
  }
}

class FocusStacker {
    
    // Core function to process the stack
    static func process(imagePaths: [String], completion: @escaping (String?, Error?) -> Void) {
        guard imagePaths.count > 1 else {
            completion(imagePaths.first, nil)
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            // 1. Load Images
            var images: [CIImage] = []
            for path in imagePaths {
                let url = URL(fileURLWithPath: path)
                if let image = CIImage(contentsOf: url) {
                    images.append(image)
                }
            }
            
            guard images.count == imagePaths.count else {
                DispatchQueue.main.async { completion(nil, NSError(domain: "FocusStacker", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load images"])) }
                return
            }
            
            // 2. Alignment (Registration) - Handle Lens Breathing
            let alignedImages = alignImages(images)
            
            // 3. Fusion ("Survival of the Sharpest")
            if let stackedImage = stackImages(alignedImages) {
                // 4. Save
                saveImage(stackedImage) { path in
                    DispatchQueue.main.async { completion(path, nil) }
                }
            } else {
                DispatchQueue.main.async { completion(nil, NSError(domain: "FocusStacker", code: 2, userInfo: [NSLocalizedDescriptionKey: "Stacking failed"])) }
            }
        }
    }
    
    // Phase 1: Image Alignment (Registration)
    // Uses Homographic registration to handle scale and warp (lens breathing).
    private static func alignImages(_ images: [CIImage]) -> [CIImage] {
        guard let ref = images.first else { return images }
        var resultImages: [CIImage] = [ref]
        saveDebugImage(ref, "debug_aligned_0.jpg")
        
        let requestHandler = VNSequenceRequestHandler()
        
        for i in 1..<images.count {
            let targetImage = images[i]
            let request = VNHomographicImageRegistrationRequest(targetedCIImage: ref, options: [:])
            
            var processedImage = targetImage
            
            do {
                try requestHandler.perform([request], on: targetImage)
                if let observation = request.results?.first as? VNImageHomographicAlignmentObservation {
                    // Vision gives a warp transform in normalized coordinates
                    if let aligned = applyHomography(targetImage, observation.warpTransform) {
                        processedImage = aligned
                    }
                }
            } catch {
                print("Alignment error image \(i): \(error)")
            }
            saveDebugImage(processedImage, "debug_aligned_\(i).jpg")
            resultImages.append(processedImage)
        }
        return resultImages
    }
    
    private static func applyHomography(_ image: CIImage, _ matrix: matrix_float3x3) -> CIImage? {
        let extent = image.extent
        let w = extent.width
        let h = extent.height
        
        // Vision coordinates are normalized (0...1) with bottom-left origin.
        // corners: BL, BR, TL, TR
        // Note: We use the normalized coordinates (0,0), (1,0), (0,1), (1,1)
        // to determine where the corners of the 'image' should land in the Reference space.
        let normalizedCorners = [
            simd_float3(0, 0, 1), // BL
            simd_float3(1, 0, 1), // BR
            simd_float3(0, 1, 1), // TL
            simd_float3(1, 1, 1)  // TR
        ]
        
        let trCorners = normalizedCorners.map { pt -> CGPoint in
            // Apply the matrix (Reference = M * Target)
            let res = simd_mul(matrix, pt)
            
            // Convert back to Homogeneous 2D
            let xNorm = CGFloat(res.x / res.z)
            let yNorm = CGFloat(res.y / res.z)
            
            // Denormalize to pixel coordinates
            // (Assumes Reference image has same dimensions as Target, which is true for burst captures)
            return CGPoint(
                x: xNorm * w + extent.origin.x,
                y: yNorm * h + extent.origin.y
            )
        }
        
        return image.applyingFilter("CIPerspectiveTransform", parameters: [
            "inputBottomLeft": CIVector(cgPoint: trCorners[0]),
            "inputBottomRight": CIVector(cgPoint: trCorners[1]),
            "inputTopLeft": CIVector(cgPoint: trCorners[2]),
            "inputTopRight": CIVector(cgPoint: trCorners[3])
        ])
    }
    
    // Phase 3: Image Fusion (Stacking)
    // "Survival of the Sharpest": f_final(x,y) = f_i(x,y) where S_i is max.
    private static func stackImages(_ images: [CIImage]) -> CIImage? {
        guard !images.isEmpty else { return nil }
        
        // Initialize composite and max map with the first image
        var composite = images[0]
        var maxSharpnessMap = getSharpnessMap(images[0])
        
        saveDebugImage(maxSharpnessMap, "debug_sharpness_0.jpg")
        
        for i in 1..<images.count {
            let nextImg = images[i]
            // Phase 2: Sharpness Calculation
            let nextSharpness = getSharpnessMap(nextImg)
            saveDebugImage(nextSharpness, "debug_sharpness_\(i).jpg")
            
            // Compare and Select Pxiel
            composite = pixelSelect(composite, nextImg, maxSharpnessMap, nextSharpness)
            
            // Update Max Map
            maxSharpnessMap = maxMap(maxSharpnessMap, nextSharpness)
        }
        
        // Save Max Sharpness Heatmap
        // Blue = Low Sharpness, Red = High Sharpness
        let heatmap = maxSharpnessMap.applyingFilter("CIFalseColor", parameters: [
            "inputColor0": CIColor(color: .blue),
            "inputColor1": CIColor(color: .red)
        ])
        saveDebugImage(heatmap, "debug_max_sharpness_heatmap.jpg")
        
        // Phase 4: Crop Artifacts
        return composite.cropped(to: images[0].extent)
    }
    
    private static func getSharpnessMap(_ image: CIImage) -> CIImage {
        // 1. Grayscale
        let lum = image.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0])
        
        // 2. Pre-smooth (Denoise)
        // Small radius to remove ISO noise but keep edges
        let smoothed = lum.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 1.0])
        
        // 3. Sobel Gradients (Robust Edge Detection)
        // Using Sobel X and Y kernels provides a better "Focus Measure" than simple Laplacian
        // as described in the reference repository's depth map generation.
        let sobelX = CIVector(values: [-1, 0, 1, -2, 0, 2, -1, 0, 1], count: 9)
        let sobelY = CIVector(values: [1, 2, 1, 0, 0, 0, -1, -2, -1], count: 9)
        
        let gradX = smoothed.applyingFilter("CIConvolution5X5", parameters: [
            kCIInputWeightsKey: sobelX,
            kCIInputBiasKey: 0.5
        ])
        
        let gradY = smoothed.applyingFilter("CIConvolution5X5", parameters: [
            kCIInputWeightsKey: sobelY,
            kCIInputBiasKey: 0.5
        ])
        
        // 4. Gradient Magnitude (Energy)
        // Calculate |Gx| + |Gy| (Approx of Magnitude)
        // We subtract 0.5 because of the bias added in convolution
        let magnitudeKernel = CIColorKernel(source: """
            kernel vec4 magnitude(__sample gx, __sample gy) {
                float dx = abs(gx.r - 0.5);
                float dy = abs(gy.r - 0.5);
                float mag = dx + dy;
                return vec4(mag, mag, mag, 1.0);
            }
        """)!
        
        let energy = magnitudeKernel.apply(extent: lum.extent, arguments: [gradX, gradY]) ?? lum
        
        // 5. Consistency Smoothing (Region consistency)
        // Matches the "depthmap smoothing" concept.
        // Aggregates sharpness scores over a local window to ensure consistent selection.
        return energy.applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 12.0])
    }
    
    private static func pixelSelect(_ oldIm: CIImage, _ newIm: CIImage, _ oldSharp: CIImage, _ newSharp: CIImage) -> CIImage {
        let kernel = CIColorKernel(source: """
            kernel vec4 select(__sample oldIm, __sample newIm, __sample oldSharp, __sample newSharp) {
                // If new pixel is sharper, use it. Otherwise keep old.
                return (newSharp.r > oldSharp.r) ? newIm : oldIm;
            }
        """)!
        return kernel.apply(extent: oldIm.extent, arguments: [oldIm, newIm, oldSharp, newSharp]) ?? oldIm
    }
    
    private static func maxMap(_ a: CIImage, _ b: CIImage) -> CIImage {
        // Keep the brighter (higher sharpness value) of the two maps
        return a.applyingFilter("CILightenBlendMode", parameters: [kCIInputBackgroundImageKey: b])
    }
    
    private static func saveDebugImage(_ image: CIImage, _ filename: String) {
        let ctx = CIContext()
        let path = NSTemporaryDirectory() + filename
        let url = URL(fileURLWithPath: path)
        if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) {
            try? ctx.writeJPEGRepresentation(of: image, to: url, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.8])
            print("DEBUG SAVED: \(path)")
            
            // Save to Photos Album
            if let cgImage = ctx.createCGImage(image, from: image.extent) {
                let uiImage = UIImage(cgImage: cgImage)
                UIImageWriteToSavedPhotosAlbum(uiImage, nil, nil, nil)
            }
        }
    }
    
    private static func saveImage(_ image: CIImage, completion: @escaping (String?) -> Void) {
        let ctx = CIContext()
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { completion(nil); return }
        let path = NSTemporaryDirectory() + UUID().uuidString + ".jpg"
        let url = URL(fileURLWithPath: path)
        do {
            try ctx.writeJPEGRepresentation(of: image, to: url, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9])
            completion(path)
        } catch {
            print("Save error: \(error)")
            completion(nil)
        }
    }
}
