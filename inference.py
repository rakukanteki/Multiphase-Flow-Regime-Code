"""
Main inference script for video retrieval
"""
import os
from config import *
from inference.retrieval import VideoRetrievalSystem


def main():
    """Main inference pipeline"""
    print("="*70)
    print(" VIDEO RETRIEVAL SYSTEM FOR FLOW REGIME CLASSIFICATION")
    print("="*70)
    
    try:
        # Initialize retrieval system
        retrieval_system = VideoRetrievalSystem(
            model_path=MODEL_PATH,
            scalers_path=SCALERS_PATH,
            video_library_csv=VIDEO_LIBRARY_CSV,
            device=DEVICE
        )
        
        # Precompute video embeddings
        retrieval_system.compute_video_library_embeddings(use_actual_pressure=False)
        
        # Define test files
        test_files = [
            os.path.join(TEST_DATA_DIR, "SlugFlow-Vsl=1.1-Vsg=0.29.xlsx"),
            os.path.join(TEST_DATA_DIR, "DispersedFlow-Vsl=2.88-Vsg=0.05.xlsx")
        ]
        
        # Process each test file
        all_results = []
        for test_file in test_files:
            if os.path.exists(test_file):
                results, retrieved_videos = retrieval_system.run_retrieval(
                    test_file, 
                    top_k=TOP_K,
                    show_frames=True
                )
                
                if results is not None:
                    all_results.append({
                        'test_file': test_file,
                        'results': results,
                        'retrieved_videos': retrieved_videos
                    })
                
                print("\n" + "="*70 + "\n")
            else:
                print(f"❌ Test file not found: {test_file}\n")
        
        print("\n" + "="*70)
        print("✅ Video retrieval completed successfully!")
        print("="*70)
        
        # Summary
        if len(all_results) > 0:
            print(f"\n📊 SUMMARY:")
            print(f"   Processed {len(all_results)} test files")
            for result_data in all_results:
                filename = os.path.basename(result_data['test_file'])
                pred_class = result_data['results']['predicted_class_name']
                confidence = result_data['results']['class_probabilities'][result_data['results']['predicted_class']]
                n_retrieved = len(result_data['retrieved_videos'])
                print(f"\n   {filename}:")
                print(f"      Predicted: {pred_class} (Conf: {confidence:.3f})")
                print(f"      Retrieved: {n_retrieved} videos")
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()