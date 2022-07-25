using TensorOperations
using Arpack
using LinearAlgebra

function ConstructBaseTensorSingle(Beta) :: Tuple{Array{Float64},Array{Float64}}
        sx = Float64[0.0 1.0;
                1.0  0.0]
        sz = Float64[1.0 0.0;
                0.0  -1.0]
        #  Base Tensor is defined according to below relation

        #           √Q
        #           |
        #       √Q——g——√Q
        #           |
        #           √Q
        Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
        g = zeros(2,2,2,2,2,2)
        g[1,1,1,1,1,1] = g[2,2,2,2,2,2] = 1
        gp =  zeros(2,2,2,2,2,2)
        gp[1,1,1,1,1,1] = -1;gp[2,2,2,2,2,2] = 1
        @tensor T[-1,-2,-3,-4,-5,-6] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*sqrt(Q)[-5,5]*sqrt(Q)[-6,6]*g[1,2,3,4,5,6]
        @tensor Tsx[-1,-2,-3,-4,-5,-6] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*sqrt(Q)[-5,5]*sqrt(Q)[-6,6]*gp[1,2,3,4,5,6]
        return T,Tsx
end


A,Asx = ConstructBaseTensorSingle(1.0)


function eigenorder(A::Array{Float64})
        F = eigen(A)
        order = sortperm(F.values,rev=true,by=abs)
        return F.values[order],F.vectors[:,order]
end



function ConstructProjector(T1::Array{Float64},T2::Array{Float64},chimax::Int64)

        lk1 = size(T1,4);lk2 = size(T2,4)

        @tensor TEMP[-1,-2,-3,-4] := T1[8,6,7,-2,9,5]*T1[8,6,7,-4,10,5]*T2[4,3,9,-1,2,1]*T2[4,3,10,-3,2,1]
        TEMP = reshape(TEMP,lk1*lk2,lk1*lk2)
        eigvalues,eigvectors = eigenorder((TEMP+TEMP')/2)
        Pf = eigvectors
        
        @tensor TEMP[-1,-2,-3,-4] := T1[-1,1,10,4,2,3]*T1[-3,1,9,4,2,3]*T2[-2,5,7,8,10,6]*T2[-4,5,7,8,9,6]
        TEMP = reshape(TEMP,lk1*lk2,lk1*lk2)
        eigvalues,eigvectors = eigenorder((TEMP+TEMP')/2)
        Pu = eigvectors
        if lk1*lk2 > chimax
                Pf = reshape(Pf,lk1,lk2,lk1*lk2)[1:lk1,1:lk2,1:chimax]
                Pu = reshape(Pu,lk1,lk2,lk1*lk2)[1:lk1,1:lk2,1:chimax]
        else    
                Pf = reshape(Pf,lk1,lk2,lk1*lk2)
                Pu = reshape(Pu,lk1,lk2,lk1*lk2)
        end
        return Pf,Pu
end


AMATRIX = [deepcopy(A) for j in 1:5]
Pf,Pu = ConstructProjector(AMATRIX[1],AMATRIX[3],10)

@tensor AMATRIX[1][-1,-2,-3,-4] := AMATRIX[2][1,9,3,7,-5,5]*AMATRIX[4][2,8,-3,6,3,4]*Pf[7,6,-4]*Pf[9,8,-2]*Pu[1,2,-1]*Pu[5,4,-6]
@tensor AMATRIX[4][-1,-2,-3,-4] := AMATRIX[1][1,9,3,7,-5,5]*AMATRIX[3][2,8,-3,6,3,4]*Pf[7,6,-4]*Pf[9,8,-2]*Pu[1,2,-1]*Pu[5,4,-6]



