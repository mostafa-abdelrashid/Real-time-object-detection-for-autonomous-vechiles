import cv2
from ultralytics import YOLO
import time
import os

class AutonomousVehicleTester:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.results = {}
        print("🚗 AUTONOMOUS VEHICLE TESTING SUITE")
        print("=" * 50)
        
    def test_webcam(self):
        """Real-time object detection with webcam"""
        print("\n🎯 Starting Webcam Test...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set optimized resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Webcam opened (640x480)")
        print("📹 Starting real-time detection...")
        print("🎯 Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        fps_history = []
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run object detection with optimized settings
            if frame_count % 2 == 0:  # Process every 2nd frame for speed
                results = self.model(frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                fps_history.append(fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)
            else:
                avg_fps = 0
            
            # Display information
            cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Autonomous Vehicle Detection", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' for screenshot", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow('Webcam Test - Autonomous Vehicle', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"webcam_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"✅ Screenshot saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Webcam Test Summary:")
        print(f"   • Total frames processed: {frame_count}")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   • Test duration: {elapsed_time:.1f} seconds")
        print(f"   • Screenshots saved: {screenshot_count}")
        print("✅ Webcam test completed!")
    
    def test_single_video(self, video_path, output_dir="output_videos"):
        """Test object detection on a single video file"""
        print(f"\n🎯 Starting Video Test: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   • Original FPS: {fps:.1f}")
        print(f"   • Total frames: {total_frames}")
        print(f"   • Duration: {duration:.1f} seconds")
        print("   • Processing video...")
        
        # Setup video writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_filename = f"detected_{os.path.basename(video_path).split('.')[0]}.avi"
        output_path = os.path.join(output_dir, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Processing variables
        frame_count = 0
        start_time = time.time()
        total_detections = 0
        fps_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run object detection with optimized settings
            results = self.model(frame, conf=0.5, verbose=False, imgsz=640)
            annotated_frame = results[0].plot()
            
            # Count detections
            if len(results) > 0:
                num_detections = len(results[0].boxes)
                total_detections += num_detections
            
            # Calculate processing FPS
            frame_count += 1
            current_time = time.time()
            processing_fps = frame_count / (current_time - start_time) if (current_time - start_time) > 0 else 0
            fps_history.append(processing_fps)
            
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_processing_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            # Show progress occasionally
            if frame_count % 60 == 0:
                cv2.imshow('Video Processing - Press ESC to skip', annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            # Write frame to output video
            out.write(annotated_frame)
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        real_time_factor = (total_frames / fps) / processing_time if processing_time > 0 else 0
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0
        
        # Store results
        video_name = os.path.basename(video_path)
        self.results[video_name] = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': avg_detections_per_frame,
            'processing_fps': avg_processing_fps,
            'processing_time': processing_time,
            'real_time_factor': real_time_factor,
            'output_video': output_path
        }
        
        # Print results
        print(f"\n📊 Video Test Results:")
        print(f"   ✅ Processed {frame_count}/{total_frames} frames")
        print(f"   🎯 Total detections: {total_detections}")
        print(f"   📈 Avg detections/frame: {avg_detections_per_frame:.1f}")
        print(f"   ⚡ Processing FPS: {avg_processing_fps:.1f}")
        print(f"   ⏱️  Processing time: {processing_time:.1f}s")
        print(f"   🚀 Real-time factor: {real_time_factor:.2f}x")
        print(f"   💾 Output saved: {output_path}")
        
        # Performance assessment
        if avg_processing_fps >= 30:
            performance = "✅ EXCELLENT"
        elif avg_processing_fps >= 15:
            performance = "👍 GOOD"
        elif avg_processing_fps >= 10:
            performance = "⚠️ ACCEPTABLE"
        else:
            performance = "❌ SLOW"
        print(f"   🎯 Performance: {performance}")
        
        return self.results[video_name]
    
    def test_multiple_videos(self, video_folder):
        """Test object detection on all videos in a folder"""
        print(f"\n🎯 Testing all videos in: {video_folder}")
        
        if not os.path.exists(video_folder):
            print(f"❌ Video folder not found: {video_folder}")
            return
        
        video_files = [f for f in os.listdir(video_folder) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"❌ No video files found in {video_folder}")
            return
        
        print(f"🔍 Found {len(video_files)} videos")
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'='*50}")
            print(f"📹 Processing video {i}/{len(video_files)}: {video_file}")
            video_path = os.path.join(video_folder, video_file)
            self.test_single_video(video_path)
    
    def generate_report(self):
        """Generate a summary report of all tests"""
        if not self.results:
            print("\n❌ No video test results available")
            return
        
        print("\n" + "="*60)
        print("📈 AUTONOMOUS VEHICLE TESTING REPORT")
        print("="*60)
        
        for video_name, metrics in self.results.items():
            print(f"\n🎬 {video_name}:")
            print(f"   • Frames processed: {metrics['total_frames']}")
            print(f"   • Total detections: {metrics['total_detections']}")
            print(f"   • Processing FPS: {metrics['processing_fps']:.1f}")
            print(f"   • Real-time capable: {'✅ YES' if metrics['real_time_factor'] >= 1.0 else '⚠️ NO'} ({metrics['real_time_factor']:.2f}x)")
            
            if metrics['processing_fps'] >= 30:
                print("   • Real-time performance: ✅ EXCELLENT")
            elif metrics['processing_fps'] >= 15:
                print("   • Real-time performance: 👍 GOOD")
            elif metrics['processing_fps'] >= 10:
                print("   • Real-time performance: ⚠️ ACCEPTABLE")
            else:
                print("   • Real-time performance: ❌ LIMITED")

def main():
    """Main menu system"""
    print("🚗 AUTONOMOUS VEHICLE DETECTION TESTER")
    print("=" * 50)
    
    # Initialize tester
    tester = AutonomousVehicleTester('best.pt')
    
    while True:
        print("\n🎯 CHOOSE TEST MODE:")
        print("1. 🎥 Webcam Test (Real-time)")
        print("2. 📹 Single Video Test")
        print("3. 📁 Multiple Videos Test (Folder)")
        print("4. 📊 Generate Report")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            tester.test_webcam()
            
        elif choice == "2":
            video_path = input("Enter video file path: ").strip()
            tester.test_single_video(video_path)
            
        elif choice == "3":
            folder_path = input("Enter folder path containing videos: ").strip()
            tester.test_multiple_videos(folder_path)
            
        elif choice == "4":
            tester.generate_report()
            
        elif choice == "5":
            print("👋 Thank you for using Autonomous Vehicle Tester!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()