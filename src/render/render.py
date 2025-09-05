import torch

def run_network(inputs, fn, netchunk):
    """
    Prepares inputs and applies network "fn" with memory optimization.
    """
    uvt_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    output_chunks = []
    
    # Process in chunks with memory management
    for i in range(0, uvt_flat.shape[0], netchunk):
        chunk = uvt_flat[i:i + netchunk]
        chunk_output = fn(chunk)
        output_chunks.append(chunk_output)
        
        # Periodic memory cleanup for large datasets
        if i > 0 and i % (netchunk * 4) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    out_flat = torch.cat(output_chunks, 0)
    out = out_flat.reshape(list(inputs.shape[:-1]) + [out_flat.shape[-1]])
    
    # Clean up intermediate tensors
    del output_chunks, uvt_flat, out_flat
    
    return out