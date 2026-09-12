import sys
import os
import torch

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import ResUNetCBAM

def main():
    checkpoint_path = 'models/resunet_v1.0.0.pth'
    print(f"=== Checkpoint Verification Tool ===")
    print(f"Loading checkpoint file: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
        
    try:
        # Construct model
        print("Constructing ResUNetCBAM model architecture...")
        model = ResUNetCBAM(in_channels=1, out_channels=1)
        model.eval()
        
        # Load state dict
        print("Loading weights from PyTorch checkpoint...")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Check if state_dict is wrapped under 'state_dict' or direct dict
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n--- CHECKPOINT COMPATIBILITY RESULTS ---")
        print(f"Missing Keys Count: {len(missing_keys)}")
        if missing_keys:
            print(f"  Missing Keys: {missing_keys[:10]}")
            
        print(f"Unexpected Keys Count: {len(unexpected_keys)}")
        if unexpected_keys:
            print(f"  Unexpected Keys: {unexpected_keys[:10]}")
            
        load_success = (len(missing_keys) == 0 and len(unexpected_keys) == 0)
        print(f"Checkpoint Loaded Successfully: {load_success}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        
        # Small dummy forward pass
        print("\n--- DUMMY FORWARD PASS VERIFICATION ---")
        dummy_input = torch.randn(1, 1, 256, 256)
        print(f"Dummy Input Shape: {list(dummy_input.shape)}")
        
        with torch.no_grad():
            dummy_output = model(dummy_input)
            
        print(f"Dummy Output Shape: {list(dummy_output.shape)}")
        print("Forward Pass Status: SUCCESS")
        
        return load_success, total_params, list(dummy_input.shape), list(dummy_output.shape)
        
    except Exception as e:
        print(f"\nFATAL ERROR during checkpoint verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
