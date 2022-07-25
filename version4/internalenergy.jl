
function InternalEnergyTreeRG(RGTensor::Array{Array{Float64}})
    chi = size(RGTensor[4],1)
    cl = RGTensor[2] ; cr = RGTensor[3]; t = RGTensor[1]
    t1 = RGTensor[4];t2 = RGTensor[5] ; t1e = RGTensor[8]
    #
    eigvalup,eigvecsup = eigsolve(y->ApplyVT(RGTensor[4],y),rand(chi,chi),1,:LM,maxiter=300)
    eigvaldn,eigvecsdn = eigsolve(y->ApplyVT(permutedims(RGTensor[4],[3,2,1,4]),y),rand(chi,chi),1,:LM,maxiter=300)
    @tensor Numerator1[] := real(eigvecsup)[1][1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[8][3,2,5,4]
    @tensor Denominator1[] := real(eigvecsup[1])[1,3]*real(eigvecsdn)[1][6,5]*RGTensor[4][1,4,6,2]*RGTensor[4][3,2,5,4]
    #
    #
    @tensor Numerator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1e[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    @tensor Denominator[] := t[17,3,5,1]*t[19,2,7,3]*t[15,11,17,9]*t[16,10,19,11]*t1[18,1,4,2]*t1[12,9,18,10]*
                            t2[5,8,15,6]*t2[7,14,16,8]*cl[4,6,13]*cr[12,14,13]
    return Numerator[1]/Denominator[1],Numerator1[1]/Denominator1[1]
end

#--------------------------------
function ApplyT(T::Array{Float64},v::Array{Float64})
    @tensor v[-1] := T[1,2,-1,2]*v[1]
    return v
end

#------------------ RhoUp contract all A and isometries up certain isometries
function ConstructRhoUp(Proj::Array{Array{Float64}},RGTensor::Array{Array{Float64}},Tsx::Array{Float64},
                         Hamiltonian::Array{Float64},eigenup::Array{Float64},eigendn::Array{Float64};direction="HORIZONTAL")
    rhoUp = Array{Array{Float64}}(undef,0)
    sitetot = size(Proj,1) + 1
    if direction == "HORIZONTAL"
        T = RGTensor[1]
    elseif direction == "VERTICAL"
        @tensor T[-1,-2,-3,-4,-5,-6] := RGTensor[1][-1,-2,1,-5]*RGTensor[1][1,-3,-4,-6]
        T = reshape(T,2,4,2,4)
    end
    Proj = reverse(Proj)
    @tensor Denominator[-1,-2,-3,-4] := eigenup[1]*eigendn[6]*Proj[1][2,3,1]*Proj[1][7,9,6]*
                        Proj[2][-1,4,2]*Proj[2][-3,8,7]*T[4,5,8,-2]*T[3,-4,9,5]
    #@tensor Denominator[-1,-2,-3,-4] := Proj[1][2,3,1]*Proj[1][7,9,1]*
    #                    Proj[2][-1,4,2]*Proj[2][-3,8,7]*T[4,5,8,-2]*T[3,-4,9,5]
    Denominator = Denominator/maximum(Denominator)
    append!(rhoUp,[Denominator])
    for j in 3:size(Proj,1)-1
        if j%2 == 1
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,-2,4,2]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,-4,5,2]
        else
            @tensor Denominator[-1,-2,-3,-4] := Denominator[1,2,4,-4]*Proj[j][-1,3,1]*Proj[j][-3,5,4]*T[3,2,5,-2]
        end
        Denominator = Denominator/maximum(Denominator)
        append!(rhoUp,[Denominator])
    end
    Proj = reverse(Proj)
    return rhoUp
end


#------ RhoDn is the contraction of A and isometries under some certain isometries
function  ConstructRhoDn(PvMatrix::Array{Array{Float64}},RGTensor::Array{Array{Float64}},Tsx::Array{Float64},
                            Hamiltonian::Array{Float64},eigenup::Array{Float64},eigendn::Array{Float64};direction="HORIZONTAL")
    rhoDn = Array{Array{Float64}}(undef,0)
    sitetot = size(PvMatrix,1) + 1
    if direction == "HORIZONTAL"
        T = RGTensor[1]
        @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*T[-2,1,-5,-6]
        TT = reshape(TT,4,2,4,2)
        TT = T
    elseif direction == "VERTICAL"
        @tensor T[-1,-2,-3,-4,-5,-6] := RGTensor[1][-1,-2,1,-5]*RGTensor[1][1,-3,-4,-6]
        T = reshape(T,2,4,2,4)
        @tensor TT[-1,-2,-3,-4,-5,-6,-7,-8] := RGTensor[1][-1,-3,2,1]*RGTensor[1][-2,1,4,-7]*
                        RGTensor[1][2,-4,-5,3]*RGTensor[1][4,3,-6,-8]
        TT = reshape(TT,4,4,4,4)
    end

    @tensor Denominator[-1,-2,-3,-4] := PvMatrix[1][3,1,-1]*PvMatrix[1][5,4,-3]*T[1,-2,4,2]*TT[3,2,5,-4]
    Denominator= Denominator/maximum(Denominator)
    append!(rhoDn,[Denominator])
    for j in 2:size(PvMatrix,1)-1
        if j %2 ==1
            @tensor Denominator[-1,-2,-3,-4] := Denominator[2,3,5,-4]*T[1,-2,4,3]*PvMatrix[j][2,1,-1]*PvMatrix[j][5,4,-3]
        elseif j%2 == 0
            @tensor Denominator[-1,-2,-3,-4] := Denominator[2,-2,4,3]*T[1,3,5,-4]*PvMatrix[j][2,1,-1]*PvMatrix[j][4,5,-3]
        end
        Denominator = Denominator/maximum(Denominator)
        append!(rhoDn,[Denominator])
    end
    return rhoDn
end



function ComputeInternalEnergyTree(rhoUp::Array{Array{Float64}},rhoDn::Array{Array{Float64}},PvMatrix::Array{Array{Float64}},
                            T::Array{Float64},Tsx::Array{Float64},site::Int64;direction = "HORIZONTAL")
    #
    if direction == "HORIZONTAL"
        T = T
        TsxL = Tsx
        TsxR = Tsx
        @tensor TT[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
        #TT = reshape(TT,4,2,4,2)
        TT = T
    elseif direction == "VERTICAL"
        #T = permutedims(T,[4,1,2,3])
        #Tsx = permutedims(Tsx,[4,1,2,3])
        @tensor TT[-1,-2,-3,-4,-5,-6,-7,-8] := T[-1,-3,2,1]*T[-2,1,4,-7]*
                T[2,-4,-5,3]*T[4,3,-6,-8]
        TT = reshape(TT,4,4,4,4)
        @tensor T[-1,-2,-3,-4,-5,-6] := T[-1,-2,1,-5]*T[1,-3,-4,-6]
        @tensor Tsx[-1,-2,-3,-4,-5,-6] := Tsx[-1,-2,1,-5]*Tsx[1,-3,-4,-6]
        T = reshape(T,2,4,2,4); Tsx = reshape(Tsx,2,4,2,4)
        TsxL = Tsx; TsxR = T

    end
    NumLayer = div(size(PvMatrix,1),2)
    PvMatrix = reverse(PvMatrix)
    if site < NumLayer -1
         @tensor Numerator[] := rhoUp[2*site-2][1,2,4,7]*PvMatrix[2*site][6,3,1]*PvMatrix[2*site][9,5,4]*PvMatrix[2*site+1][11,8,6]*
                        PvMatrix[2*site+1][15,10,9]*PvMatrix[2*site+2][16,13,11]*PvMatrix[2*site+2][18,14,15]*rhoDn[2*NumLayer-2*site-2][16,17,18,19]*
                        TsxL[3,2,5,12]*TsxR[13,12,14,17]*T[8,19,10,7]
        @tensor Denominator[] :=  rhoUp[2*site-2][1,2,4,7]*PvMatrix[2*site][6,3,1]*PvMatrix[2*site][9,5,4]*PvMatrix[2*site+1][11,8,6]*
                        PvMatrix[2*site+1][15,10,9]*PvMatrix[2*site+2][16,13,11]*PvMatrix[2*site+2][18,14,15]*rhoDn[2*NumLayer-2*site-2][16,17,18,19]*
                        T[3,2,5,12]*T[13,12,14,17]*T[8,19,10,7]
    elseif site == NumLayer-1
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*TsxL[3,2,5,12]*TsxR[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TT[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site == NumLayer
        #
        if direction == "HORIZONTAL"
            @tensor TsxR[-1,-2,-3,-4,-5,-6] :=Tsx[-1,-3,-4,1]*T[-2,1,-5,-6]
            TsxR = reshape(TsxR,4,2,4,2)
            @tensor TR[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*T[-2,1,-5,-6]
            TR = reshape(TR,4,2,4,2)
            TsxR = Tsx;TR = T
        elseif direction == "VERTICAL"
            TsxL = Tsx;
            @tensor TR[-1,-2,-3,-4,-5,-6,-7,-8] := RGTensor[1][-1,-3,2,1]*RGTensor[1][-2,1,4,-7]*
                        RGTensor[1][2,-4,-5,3]*RGTensor[1][4,3,-6,-8]
            TR = reshape(TR,4,4,4,4)
            TsxR = TR
        end
        #
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*TsxL[13,12,15,17]*TsxR[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TR[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site == NumLayer+1
        if direction == "HORIZONTAL"
            @tensor TsxL[-1,-2,-3,-4,-5,-6] :=T[-1,-3,-4,1]*Tsx[-2,1,-5,-6]
            TsxL = reshape(TsxL,4,2,4,2)
            TsxR = Tsx
            @tensor TL[-1,-2,-3,-4,-5,-6] := T[-1,-3,-4,1]*T[-2,1,-5,-6]
            TL = reshape(TL,4,2,4,2)
            TsxL = Tsx;TL = T
        elseif  direction == "VERTICAL"
            TsxR = Tsx
            @tensor TL[-1,-2,-3,-4,-5,-6,-7,-8] := RGTensor[1][-1,-3,2,1]*RGTensor[1][-2,1,4,-7]*
                                RGTensor[1][2,-4,-5,3]*RGTensor[1][4,3,-6,-8]
            TL = reshape(TL,4,4,4,4)
            TsxL = TL
        end
        #
        @tensor Numerator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TsxL[16,17,18,19]*TsxR[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
        @tensor Denominator[] := rhoUp[end-2][1,2,4,8]*T[3,2,5,12]*T[13,12,15,17]*TL[16,17,18,19]*T[7,19,9,8]*PvMatrix[end-2][6,3,1]*PvMatrix[end-2][10,5,4]*
                        PvMatrix[end-1][11,7,6]*PvMatrix[end-1][14,9,10]*PvMatrix[end][16,13,11]*PvMatrix[end][18,15,14]
    elseif site > NumLayer+1
        @tensor Numerator[] := rhoUp[4*NumLayer-2*site-1][1,7,4,3]*rhoDn[2*(site-NumLayer-1)-1][16,17,18,19]*PvMatrix[4*NumLayer-2*site+1][6,2,1]*
                        PvMatrix[4*NumLayer-2*site+1][9,5,4]*PvMatrix[4*NumLayer-2*site+2][11,8,6]*PvMatrix[4*NumLayer-2*site+2][14,10,9]*
                        PvMatrix[4*NumLayer-2*site+3][16,12,11]*PvMatrix[4*NumLayer-2*site+3][18,15,14]*TsxL[12,19,15,13]*TsxR[2,13,5,3]*T[8,7,10,17]
        @tensor Denominator[] := rhoUp[4*NumLayer-2*site-1][1,7,4,3]*rhoDn[2*(site-NumLayer-1)-1][16,17,18,19]*PvMatrix[4*NumLayer-2*site+1][6,2,1]*
                        PvMatrix[4*NumLayer-2*site+1][9,5,4]*PvMatrix[4*NumLayer-2*site+2][11,8,6]*PvMatrix[4*NumLayer-2*site+2][14,10,9]*
                        PvMatrix[4*NumLayer-2*site+3][16,12,11]*PvMatrix[4*NumLayer-2*site+3][18,15,14]*T[12,19,15,13]*T[2,13,5,3]*T[8,7,10,17]

    end
    PvMatrix = reverse(PvMatrix)
    internalenergytree = Numerator[1]/Denominator[1]
    return internalenergytree
    #
end



function ComputeInternalEnergyTreeDouble(PvMatrix::Array{Array{Float64}},eigenup,eigendn,site::Int64,sz::Array{Float64})
    NumLayer = div(size(PvMatrix,1),2)
    PvMatrix = reverse(PvMatrix)
    if site < NumLayer -1
        @tensor TDn[-1,-2] := PvMatrix[2*site][8,7,-1]*PvMatrix[2*site][10,9,-2]*PvMatrix[2*site+1][4,6,8]*
                       PvMatrix[2*site+1][5,6,10]*PvMatrix[2*site+2][3,1,4]*PvMatrix[2*site+2][3,2,5]*sz[1,2]*sz[7,9]
        for j in 2*site-1:-1:1
            @tensor TDn[-1,-2] := PvMatrix[j][1,3,-1]*PvMatrix[j][2,3,-2]*TDn[1,2]
        end
    end

    #@tensor Numerator[] := eigenup[1]*TDn[1,2]*eigendn[2]
    #@tensor Denominator[] := eigenup[1]*eigendn[1]
    #println(TDn)
    append!(TDnMatrix,[TDn])
    Numerator = TDn[1,1]
    Denominator = 1
    return Numerator[1]/Denominator[1]
end



function ComputeInternalEnergyMPS(PvMatrix::Array{Array{Float64}},eigenup::Array{Float64},eigendn::Array{Float64},
                                site::Int64,sz::Array{Float64})
    NumLayer = size(PvMatrix,1)
    PvMatrix = reverse(PvMatrix)
    @tensor TDn[-1,-2] := PvMatrix[site][5,4,-1]*PvMatrix[site][7,6,-2]*PvMatrix[site+1][3,1,5]*
                            PvMatrix[site+1][3,2,7]*sz[1,2]*sz[4,6]
    for j in site-1:-1:1
        @tensor TDn[-1,-2] := PvMatrix[j][1,2,-1]*PvMatrix[j][3,2,-2]*TDn[1,3]
    end

    @tensor Numerator[] := eigenup[1]*eigendn[2]*TDn[1,2]
    @tensor Denominator[] :=eigenup[1]*eigendn[1]
    return Numerator[1]/Denominator[1]
end
