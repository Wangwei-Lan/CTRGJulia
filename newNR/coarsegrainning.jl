

function CorseGraining(T::Array{Float64},NumLayer::Int64,chimax;Tsx=Float64[1.0 0.0; 0.0 -1.0])
    println("start corse Graining")

    #---- Diagonal corner tensor
    TL,TR = svdT(T)
    #TU,TD = svdT(permutedims(T,[4,1,2,3]))
    TL = TU; TD = TR
    #--- Boundary tensor
    # RGTensor[1]   base tensor T
    # RGTensor[2]   Center Tensor Left : cl
    # RGTensor[3]   Center Tensor Right : cr
    # RGTensor[4]   vertical up tensor : vtu
    # RGTensor[5]   vertical dn tensor : vtd
    # RGTensor[6]   horizontal left tensor : htl
    # RGTensor[7]   horizontal right tensor : htr
    RGTensor = Array{Array{Float64}}(undef,8)
    RGTensor[1] = T
    RGTensor[2] = TL;RGTensor[3] = TR
    RGTensor[4:end] = [T for j in 4:7]
    RGTensor[8] = Tsx
    #-- Norms of different tensor
    RGnorm = Array{Array{Float64}}(undef,7)
    RGnorm = [[] for j in 1:7]

    NumSite = Array{Int64}(undef,0)
    FEttrg = Array{Float64}(undef,0)
    FEttrg1 = Array{Float64}(undef,0)
    println("start loop")
    for j in 1:NumLayer
        println("This is NumLayer $j")

        @time begin
        #---- compute projectors
        @time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="LD")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        RGTensor[5][5,1,3,6]*TL[3,2,7]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        RGTensor[4][5,3,1,6]*TR[2,3,7]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][-1,4,1,2]*T[1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        if j == 1
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*Tsx[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        else
            @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][3,1,5,-4]*T[2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        end
        end
        NormalizeTensor(RGTensor,RGnorm)

        @time begin
        PcL,PcR,Ph,Pv = Mover(RGTensor,TL,TR,chimax,direction="RU")
        @tensor RGTensor[2][-1,-2,-3] := RGTensor[2][6,4,7]*TL[3,2,8]*RGTensor[4][1,5,6,2]*
                                    Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
        @tensor RGTensor[3][-1,-2,-3] := RGTensor[3][4,5,7]*TR[2,3,8]*RGTensor[5][2,5,6,1]*
                                    Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
        @tensor RGTensor[5][-1,-2,-3,-4] := RGTensor[5][1,5,-3,3]*T[-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
        @tensor RGTensor[4][-1,-2,-3,-4] := RGTensor[4][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        @tensor RGTensor[8][-1,-2,-3,-4] := RGTensor[8][2,-2,4,1]*T[3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
        end
        NormalizeTensor(RGTensor,RGnorm)
        #---- Normalize to avoid calculation explotion
        end
        #
        Ns = 1+8*sum(1:1:j)
        @tensor Z[] := RGTensor[2][1,2,3]*RGTensor[3][1,2,3]
        #println("Z ",Z[1])
        FE = (log(Z[1]) + sum(log.(RGnorm[2]))+ sum(log.(RGnorm[3]))+
                sum([j-1:-1:0...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(RGnorm[5][2:2:end])))/Ns
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        #
        #=
        Ns = 2*(2*j+1)
        Lambda = FreeEnergyDensity(RGTensor[4],RGTensor[4])
        FE = (real(Lambda)+sum(log.(RGnorm[4]))+sum(log.(RGnorm[4])))/(2*(2*j+1))
        append!(FEttrg,[FE])
        append!(NumSite,[Ns])
        =#

    end
    return RGTensor,RGnorm,FEttrg,FEttrg1,NumSite
end
