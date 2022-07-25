
#------- function ordered eigen value decomposition
function eigenorder(A::Array{Float64})
    F = eigen(A)
    order = sortperm(F.values,rev=true,by=abs)
    return F.values[order],F.vectors[:,order]
end

function ApplyAT(T::Array{Float64},A::Array{Float64},x::Array{Float64})
    @tensor x[-1,-2] := x[1,2]*T[1,4,-1,3]*A[2,3,-2,4]
    x = x/maximum(x)
    return x
end

function ApplyVT(VT::Array{Float64},v::Array{Float64})
    @tensor v[-1,-2] := VT[1,4,-1,2]*VT[3,2,-2,4]*v[1,3]
    return v
end


function svdT(T::Array{Float64})
    sizeT = size(T,1)
    T = reshape(permutedims(T,[1,2,4,3]),sizeT^2,sizeT^2)
    F = svd((T+T')/2)
    T1 = F.U*Matrix(Diagonal(sqrt.(F.S)))
    T2 = F.V*Matrix(Diagonal(sqrt.(F.S)))
    T1 = reshape(T1,sizeT,sizeT,sizeT^2)
    T2 = permutedims(reshape(T2,sizeT,sizeT,sizeT^2),[2,1,3])
    #T2 = reshape(T2,sizeT,sizeT,sizeT^2)
    return T1,T2
end




function NormalizeTensor(RGTensor,RGnorm)
    for j in 2:5
        #println(maximum(RGTensor[j]))
        append!(RGnorm[j],maximum(RGTensor[j]))
        if j == 4
            RGTensor[6] = RGTensor[6]/maximum(RGTensor[j])
        end
        RGTensor[j] = RGTensor[j]/maximum(RGTensor[j])
    end
end
