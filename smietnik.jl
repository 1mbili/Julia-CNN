
# function maxPoolB(x, g, kernel_size, cache)
#     println(g[:, :, 1, 1])
#     dupa
#     H, W, C, N = size(x)
#     Gh, Gw, Gc, Gn = size(g)
#     dx = zeros(H, W, C, N)
#     K_H = kernel_size[1]
#     K_W = kernel_size[2]
#     for n=1:Gn
#         for c=1:Gc
#             for h=1:Gh
#                 for w=1:Gw
#                     max_val = x[1+(h-1)*K_H:h*(K_H), 1+(w-1)*K_W:w*K_W, c, n]
#                     max_h, max_w = Tuple.(findmax(max_val)[2])
#                     max_h += (h-1)*K_H
#                     max_w += (w-1)*K_W
#                     dx[max_h,max_w,c,n] = g[h,w,c,n]
#                 end
#             end
#         end
#     end
#     return tuple(dx)
# end

# function maxPool(x, kernel_size, cache)
#     println(x[:, :, 1, 1])
#     aga

#     #N, C, H, W = size(x)
#     H, W, C, N = size(x)
#     K_H = kernel_size[1]
#     K_W = kernel_size[2]
#     W_2 = fld(W - K_W, K_W) + 1
#     H_2 = fld(H - K_H ,K_H) + 1
#     #out = zeros(N, C, H_2, W_2)
#     out = zeros(H_2, W_2, C, N)
#     for n=1:N
#         for c=1:C
#             for h=1:H_2
#                 @views for w=1:W_2
#                     out[h, w, c, n] = maximum(x[K_H*(h-1)+1:K_H*h, K_W*(w-1)+1:K_W*w, c, n])
#                     #val, ind = findmax(x[K_H*(h-1)+1:K_H*h, K_W*(w-1)+1:K_W*w, c, n])
#                     #out[h, w, c, n] = val
#                     #ix, iy = ind[1] + K_H*(h-1), ind[2] + K_W*(w-1) # +1 -1 sie skraca
#                     #ix, iy = ind[1] + K_H*(h-1), ind[2] + K_W*(w-1) # +1 -1 sie skraca
#                     #push!(cache, CartesianIndex(ix, iy, c, n))
#                     #push!(cache, CartesianIndex(n, c, ix, iy))
#                 end
#             end
#         end
#     end
#     return out
# end