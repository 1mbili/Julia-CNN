using Random
using NNlib: conv
using Flux
Random.seed!(1234);

function conv_nowe(I, K)
    H, W, C, N = size(I) 
    HH, WW, C, F = size(K)
    H_R = 1 + H - HH
    W_R = 1 + W - WW
    out = zeros(H_R, W_R, F, N)
    for n=1:N
        for depth=1:F
            @views for r=1:H_R
                for c=1:W_R
                    out[r, c, depth, n] = sum(I[r:r+HH-1, c:c+WW-1, :, n] .* K[:, :, :, depth]) 
                end
            end
        end
    end
    return out
end




function conv_noweP(I, K)
    #I = reshape(I, 28 , 28, 1, 1)
    H, W, C, N = size(I) 
    HH, WW, C, F = size(K)
    H_R = 1 + H - HH
    W_R = 1 + W - WW
    out = zeros(H_R, W_R, F, N)
    for r=1:H_R
        @views for c=1:W_R
            r_field = I[r:r+HH-1, c:c+WW-1, :, :]
            r_field_flat = reshape(r_field, HH*WW*C, N)
            K_flat = reshape(K, HH*WW*C, F)
            out[r, c, :, :] = sum(K_flat .* r_field_flat, dims = 1)
        end
    end
    return out
end

function create_kernels_nowe(kernel_height, kernel_width, n_input, n_output)
    # Inicjalizacja Xaviera
    squid = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
    random_vals = randn(kernel_height, kernel_width, n_input, n_output) * squid
    return random_vals
end

function create_kernels(n_input::Int64, n_output::Int64, kernel_width::Int64, kernel_height::Int64)
    # Inicjalizacja Xaviera
    squid::Float32 = sqrt(6 / (n_input + n_output * (kernel_width * kernel_height)))
    random_vals = randn(Float32, kernel_height, kernel_width, n_input, n_output) * squid
    return random_vals
end
@inline function im2col(A, n, m)
    M,N = size(A)
    B = Array{Float32}(undef, m*n, (M-m+1)*(N-n+1))
    indx = reshape(1:M*N, M,N)[1:M-m+1,1:N-n+1]
    for (i,value) in enumerate(indx)
      for j = 0:n-1
          @views B[(i-1)*m*n+j*m+1:(i-1)m*n+(j+1)m] = A[value+j*M:value+m-1+j*M]
      end
    end
    return B
   end
   
  
  function conv_fast2(I::Array{Float32, 4}, K::Array{Float32, 4})
      KH, KW, KIC, KOC = size(K)
      IH, IW, IC, IN = size(I)
      H_O = 1 + IH - KH
      W_O = 1 + IW - KW
      col_I = im2col(I, KH, KW)
      output = zeros(Float32, H_O, W_O, KOC, IN)
  
      @inbounds @fastmath @views for k in 1:KOC  # Each output channel
          col_K = transpose(reshape(K[:, :, :, k], KH * KW * KIC, 1))  # Reshape all input channel filters for this output
          conv = col_K * col_I
          conv_reshaped = reshape(conv, H_O, W_O, IN)
          output[:, :, k, :] .+= conv_reshaped  # Sum convolutions from each input channel
      end
  
      return output
  end
  
  
test_n = Float32.([0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.21568628 0.53333336 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.6745098 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.07058824 0.8862745 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.19215687 0.07058824 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.67058825 0.99215686 0.99215686 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.11764706 0.93333334 0.85882354 0.3137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09019608 0.85882354 0.99215686 0.83137256 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.14117648 0.99215686 0.99215686 0.6117647 0.05490196 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25882354 0.99215686 0.99215686 0.5294118 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.36862746 0.99215686 0.99215686 0.41960785 0.003921569 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.09411765 0.8352941 0.99215686 0.99215686 0.5176471 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.6039216 0.99215686 0.99215686 0.99215686 0.6039216 0.54509807 0.043137256 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.44705883 0.99215686 0.99215686 0.95686275 0.0627451 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.011764706 0.6666667 0.99215686 0.99215686 0.99215686 0.99215686 0.99215686 0.74509805 0.13725491 0.0 0.0 0.0 0.0 0.0 0.15294118 0.8666667 0.99215686 0.99215686 0.52156866 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.99215686 0.8039216 0.3529412 0.74509805 0.99215686 0.94509804 0.31764707 0.0 0.0 0.0 0.0 0.5803922 0.99215686 0.99215686 0.7647059 0.043137256 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7764706 0.043137256 0.0 0.007843138 0.27450982 0.88235295 0.9411765 0.1764706 0.0 0.0 0.18039216 0.8980392 0.99215686 0.99215686 0.3137255 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.07058824 0.99215686 0.99215686 0.7137255 0.0 0.0 0.0 0.0 0.627451 0.99215686 0.7294118 0.0627451 0.0 0.50980395 0.99215686 0.99215686 0.7764706 0.03529412 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49411765 0.99215686 0.99215686 0.96862745 0.16862746 0.0 0.0 0.0 0.42352942 0.99215686 0.99215686 0.3647059 0.0 0.7176471 0.99215686 0.99215686 0.31764707 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.53333336 0.99215686 0.9843137 0.94509804 0.6039216 0.0 0.0 0.0 0.003921569 0.46666667 0.99215686 0.9882353 0.9764706 0.99215686 0.99215686 0.7882353 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6862745 0.88235295 0.3647059 0.0 0.0 0.0 0.0 0.0 0.0 0.09803922 0.5882353 0.99215686 0.99215686 0.99215686 0.98039216 0.30588236 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.101960786 0.6745098 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.105882354 0.73333335 0.9764706 0.8117647 0.7137255 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.6509804 0.99215686 0.32156864 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.2509804 0.007843138 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.9490196 0.21960784 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.96862745 0.7647059 0.15294118 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.49803922 0.2509804 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;;;;]);
jj = create_kernels(1, 6, 3, 3);
@profview conv_fast2(test_n, jj)
