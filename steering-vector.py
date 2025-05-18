import torch
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Cache residuals and analyze as potential steering vector
def analyze_steering_vector(model, prompt1, prompt2, prompt_names=["Sequence 1", "Sequence 2"], test_input=None):
    """
    Analyze whether the residual difference between two prompts can act as a steering vector.
    
    Args:
        model: The transformer model to analyze
        prompt1: First prompt sequence
        prompt2: Second prompt sequence 
        prompt_names: Names of the prompts for visualization
        test_input: Optional input for testing the steering vector effect
        
    Returns:
        Dictionary containing the steering vector analysis results
    """
    model.eval()
    
    # Ensure prompts are properly formatted tensors
    prompt1 = prompt1.unsqueeze(0).to(DEVICE) if prompt1.dim() == 1 else prompt1.to(DEVICE)
    prompt2 = prompt2.unsqueeze(0).to(DEVICE) if prompt2.dim() == 1 else prompt2.to(DEVICE)
    
    # Get residual streams by running with cache
    _, cache1 = model.run_with_cache(prompt1)
    _, cache2 = model.run_with_cache(prompt2)
    
    # Extract the final residual stream from both sequences
    resid1 = cache1["resid_post", -1]  # Last layer, post-attention residual
    resid2 = cache2["resid_post", -1]
    
    # Calculate the steering vector (difference between residuals)
    steering_vector = resid2 - resid1
    
    # Compute magnitude of the steering vector
    steering_mag = torch.norm(steering_vector, dim=-1).mean().item()
    
    print(f"Steering vector shape: {steering_vector.shape}")
    print(f"Average magnitude: {steering_mag:.4f}")
    
    # Test steering vector by applying it to original prompts
    results = {}
    results["steering_vector"] = steering_vector
    results["magnitude"] = steering_mag
    
    # Create a new prompt for testing
    if test_input is None:
        test_input = torch.tensor([0] * 10).unsqueeze(0).to(DEVICE)
    # Get baseline prediction
    with torch.no_grad():
        baseline_logits = model(test_input)
        baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
        
    # Apply steering vector at different scales and get predictions
    scales = [-2.0, -1.0, -0.5, 0, 0.5, 1.0, 2.0]
    steered_results = []
    
    for scale in scales:
        with torch.no_grad():
            _, test_cache = model.run_with_cache(test_input)
            final_resid = test_cache["resid_post", -1]
            
            # Apply steering vector to the residual
            modified_resid = final_resid + scale * steering_vector
            
            # Forward pass using modified residual
            # This requires model-specific hooking, let's approximate:
            modified_logits = model.unembed(model.ln_final(modified_resid))
            modified_probs = torch.softmax(modified_logits[0, -1], dim=-1)
            
            steered_results.append({
                'scale': scale,
                'prob_0': modified_probs[0].item(),
                'prob_1': modified_probs[1].item()
            })
    
    results["steered_results"] = steered_results
    
    # Visualize the effect of steering
    plt.figure(figsize=(10, 6))
    
    scales_list = [res['scale'] for res in steered_results]
    probs_1 = [res['prob_1'] for res in steered_results]
    
    plt.plot(scales_list, probs_1, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='gray', linestyle='--', label='P=0.5')
    plt.grid(True)
    plt.xlabel('Steering Vector Scale')
    plt.ylabel('Probability of Next Token Being 1')
    plt.title(f'Effect of Applying {prompt_names[1]}-{prompt_names[0]} Steering Vector')
    plt.xticks(scales_list)
    plt.ylim(0, 1)
    plt.show()
    # # Visualize the steering vector components
    # plt.figure(figsize=(12, 6))
    # plt.plot(steering_vector[0, -1].cpu().detach().numpy())
    # plt.title(f'Steering Vector Components ({prompt_names[1]}-{prompt_names[0]})')
    # plt.xlabel('Component Index')
    # plt.ylabel('Value')
    # plt.grid(True)
    # plt.show()
    return results
