function ConstructProjectorCenter(projcenter::Array{Float64},chimax::Int64;num="double")
    Dlink = size(projcenter,1)
    chicl3 = size(projcenter,2)
    Dlink*chicl3 > chimax ? pcL = rand(Dlink,chicl3,chimax) : pcL = rand(Dlink,chicl3,Dlink*chicl3)
    slast = rand(10)
    k= 0
    for j in 1:150
        chi = size(pcL,3)
        @tensor Env[-1,-2,-3] := projcenter[1,2,3,4]*pcL[1,2,-1]*projcenter[-2,-3,3,4]
        #Env = Env/maximum(Env)
        Env = reshape(Env,chi,Dlink*chicl3)
        F = svd(Env)
        pcL = reshape(F.V*F.U',Dlink,chicl3,chi)
        if length(F.S) == length(slast)
            abs.(norm(F.S-slast)/norm(slast)) < 1.0e-13 ? break : ()
        end
        slast = F.S
        k+=1
    end
    println("isometry loop is $k")
    if num == "single"
        return pcL
    elseif num == "double"
        pcR = pcL
        return pcL,pcR
    end
end


function ConstructProjectorCenter4(projcenter::Array{Float64},chimax::Int64)
    Dlink = size(projcenter,1);chicl3 = size(projcenter,2)
    projcenter = reshape(projcenter,Dlink*chicl3,Dlink*chicl3)

    #F1 = svd(projcenter)
    eigvalues,eigvector = eigenorder(projcenter)
    pc = eigvector
    if Dlink*chicl3> chimax
        pc = reshape(pc[:,1:chimax],Dlink,chicl3,chimax)
    else
        pc = reshape(pc,Dlink,chicl3,Dlink*chicl3)
    end

    return pc
end


function ConstructProjectorCenter1(projcenter::Array{Float64},chimax::Int64)
    pcL = ConstructProjectorCenter(projcenter,chimax,num="single")
    pcR = ConstructProjectorCenter(permutedims(projcenter,[3,4,1,2]),chimax,num="single")

    @tensor mixed[-1,-2] := pcL[1,2,-1]*pcR[1,2,-2]
    F = svd(mixed)
    invS = Matrix(Diagonal(F.S))^(1/2)
    @tensor pcL[-1,-2,-3] := pcL[-1,-2,2]*F.U[2,1]*invS[1,-3]   # with invS, the internal energy will fluctuate
    @tensor pcR[-1,-2,-3] := pcR[-1,-2,2]*F.V[2,1]invS[1,-3]    # not only fluctuate, will jump to another value also
    #@tensor pcL[-1,-2,-3] := pcL[-1,-2,2]*F.U[2,-3]
    #@tensor pcR[-1,-2,-3] := pcR[-1,-2,2]*F.V[2,-3]
    return pcL,pcR
end

function ConstructProjectorCenter2(projcenter1::Array{Float64},projcenter2::Array{Float64},
                                chimax::Int64)  # seems best for now
    #pcL = ConstructProjectorCenter((projcenter1+permutedims(projcenter1,[3,4,1,2]))/2,chimax,num="single")
    #pcR = ConstructProjectorCenter((projcenter2+permutedims(projcenter2,[3,4,1,2]))/2,chimax,num="single")
    pcL = ConstructProjectorCenter4((projcenter1+permutedims(projcenter1,[3,4,1,2]))/2,chimax)
    pcR = ConstructProjectorCenter4((projcenter2+permutedims(projcenter2,[3,4,1,2]))/2,chimax)

    @tensor mixed[-1,-2] := pcL[1,2,-1]*pcR[1,2,-2]
    F = svd(mixed)
    @tensor pcL[-1,-2,-3] := pcL[-1,-2,2]*F.U[2,1]*sqrt(Matrix(Diagonal(F.S)))[1,-3] # the same results as previous,most of them working fine. but
    @tensor pcR[-1,-2,-3] := pcR[-1,-2,2]*F.V[2,1]*sqrt(Matrix(Diagonal(F.S)))[1,-3] # some of them does not work, not because fluctuate
    return pcL,pcR
end


function ConstructProjectorHorizontal(H::Array{Float64},T::Array{Float64},chimax::Int64;direction ="LD")
    Dlink = size(H,1);chi = size(H,2)
    if direction == "LD"
        @tensor Transfer[-1,-2,-3,-4] := H[2,1,5,-1]*H[2,1,6,-3]*T[5,3,4,-2]*T[6,3,4,-4]
    elseif direction == "RU"
        @tensor Transfer[-1,-2,-3,-4] := H[5,3,4,-1]*H[6,3,4,-3]*T[2,1,5,-2]*T[2,1,6,-4]
    end

    Transfer = (Transfer+permutedims(Transfer,[3,4,1,2]))/2
    Transfer = Transfer/maximum(Transfer)
    #---- multiply 100 Transfer together
    for j in 1:50
        if direction =="LD"
            #@tensor Transfer[-1,-2,-3,-4] := Transfer[1,2,-3,-4]*H[4,-1,3,1]*T[3,-2,4,2]
            @tensor Transfer[-1,-2,-3,-4] := Transfer[1,4,3,8]*H[2,1,5,-1]*H[2,3,7,-3]*T[5,4,6,-2]*T[7,8,6,-4]
            Transfer = (Transfer+permutedims(Transfer,[3,4,1,2]))/2
        elseif direction == "RU"
            #@tensor Transfer[-1,-2,-3,-4] := Transfer[1,2,-3,-4]*H[3,-1,4,1]*T[4,-2,3,2]
            @tensor Transfer[-1,-2,-3,-4] := Transfer[1,4,3,7]*H[5,1,2,-1]*H[8,3,2,-3]*T[6,4,5,-2]*T[6,7,8,-4]
            Transfer = (Transfer+permutedims(Transfer,[3,4,1,2]))/2
        end
        Transfer = Transfer/maximum(Transfer)
    end
    #@tensor Transfer[-1,-2,-3,-4] := Transfer[1,2,-1,-2]*Transfer[1,2,-3,-4]
    Transfer = reshape(Transfer,Dlink*chi,Dlink*chi)
    F = svd((Transfer+Transfer')/2)
    ph = reshape(F.U,chi,Dlink,Dlink*chi)

    chi*Dlink> chimax ? ph = ph[:,:,1:chimax] : ()


    return ph

end





function  Mover(RGTensor::Array{Array{Float64}},TL::Array{Float64},TR::Array{Float64},chimax::Int64;direction=nothing)
    # cl : Center left Tensor
    # cr : Center Right Tensor
    T = RGTensor[1]
    cl = RGTensor[2];cr = RGTensor[3]
    #  t1 : vertical up tensor
    #  t2 : vertical dn tensor
    #  t3 : horizontal left tensor
    #  t4 : horizontal right tensor
    t1 = RGTensor[4];t2 = RGTensor[5];#t3 = RGTensor[6];t4 = RGTensor[7]
    chicl1 = size(cl,1);chicl3 = size(cl,3);dlink = 2


    #------------------------------------   Method 1 // Method 1 and Method 2 are similar but only small modification
    #
    if direction == "LD"
        #@tensor projcenter[-1,-2,-3,-4] := cl[4,3,-2]*cr[5,4,-4]*TL[6,7,-1]*TR[7,8,-3]*t2[1,2,6,3]*t1[5,8,2,1]
        @tensor projcenter1[-1,-2,-3,-4] := cl[4,3,-2]*cl[4,5,-4]*TL[6,7,-1]*TL[8,7,-3]*t2[1,2,6,3]*t2[1,2,8,5]
        @tensor projcenter2[-1,-2,-3,-4] := cr[5,4,-4]*cr[3,4,-2]*TR[7,8,-3]*TR[7,6,-1]*t1[5,8,2,1]*t1[3,6,2,1]
        #
        @tensor projhorizontal[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t1[3,14,2,1]*t1[5,12,2,1]
        @tensor projvertical[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t1[9,13,-3,11]*t1[10,15,-1,11]
        #
        #Ph = ConstructProjectorHorizontal(RGTensor[5],RGTensor[1],chimax,direction="LD")
        #Pv = ConstructProjectorHorizontal(permutedims(RGTensor[4],[4,1,2,3]),permutedims(RGTensor[1],[4,1,2,3]),chimax,direction="LD")
    elseif  direction == "RU"
        #@tensor projcenter[-1,-2,-3,-4] := cl[3,4,-2]*cr[4,5,-4]*TL[6,7,-1]*TR[8,6,-3]*t1[2,1,3,7]*t2[8,5,1,2]
        @tensor projcenter1[-1,-2,-3,-4] := cl[3,4,-2]*cl[5,4,-4]*TL[6,7,-1]*TL[6,8,-3]*t1[2,1,3,7]*t1[2,1,5,8]
        @tensor projcenter2[-1,-2,-3,-4] := cr[4,5,-4]*cr[4,3,-2]*TR[8,6,-3]*TR[7,6,-1]*t2[8,5,1,2]*t2[7,3,1,2]
        #
        @tensor projvertical[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t2[13,5,2,1]*t1[-3,10,11,14]*t2[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[3, 4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t2[15,11,10,-3]*
                            t1[1,2,5,13]*t2[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
        #
        #Ph = ConstructProjectorHorizontal(RGTensor[5],RGTensor[1],chimax,direction="RU")
        #Pv = ConstructProjectorHorizontal(permutedims(RGTensor[4],[4,1,2,3]),permutedims(RGTensor[1],[4,1,2,3]),chimax,direction="RU")

    end
    PcL,PcR = ConstructProjectorCenter2(projcenter1,projcenter2,chimax)
    #
    projhorizontal = reshape(projhorizontal,chicl1*dlink,chicl1*dlink)
    #F = svd((projhorizontal+projhorizontal')/2)
    eigvalue,eigvector = eigenorder((projhorizontal+projhorizontal')/2)
    dlink*chicl1 > chimax ? Ph = reshape(eigvector[:,1:chimax],chicl1,dlink,chimax) :
                        Ph = reshape(eigvector,chicl1,dlink,dlink*chicl1)
    #
    #
    projvertical = reshape(projvertical,chicl1*dlink,chicl1*dlink)
    #F = svd((projvertical+projvertical')/2)
    eigvalue,eigvector = eigenorder((projvertical+projvertical')/2)
    dlink*chicl1 > chimax ? Pv = reshape(eigvector[:,1:chimax],chicl1,dlink,chimax) :
                            Pv= reshape(eigvector,chicl1,dlink,dlink*chicl1)
    #

    #-------------------------------------- Method 2 // just a slightly modification of Method 1, just a test
    #=
    if direction == "LD"
        @tensor projhorizontal[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t1[3,14,2,1]*t1[5,12,2,1]
        @tensor projvertical[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t1[9,13,-3,11]*t1[10,15,-1,11]
    elseif  direction == "RU"
        @tensor projvertical[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t2[13,5,2,1]*t1[-3,10,11,14]*t2[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor projhorizontal[-1,-2,-3,-4] := cl[3,4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t2[15,11,10,-3]*
                            t1[1,2,5,13]*t2[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
    end
    projhorizontal = reshape(projhorizontal,chicl1*dlink,chicl1*dlink)
    F = svd((projhorizontal+projhorizontal')/2)
    dlink*chicl1 > chimax ? Ph = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                        Ph = reshape(F.U,chicl1,dlink,dlink*chicl1)
    projvertical = reshape(projvertical,chicl1*dlink,chicl1*dlink)
    F = svd((projvertical+projvertical')/2)
    dlink*chicl1 > chimax ? Pv = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                            Pv= reshape(F.U,chicl1,dlink,dlink*chicl1)
    #
    if direction == "LD"
        @tensor projcenter1[-1,-2,-3,-4] := cl[4,6,-2]*cl[10,11,-4]*t2[5,2,1,6]*t2[12,9,7,11]*
                            TL[1,3,-1]*TL[7,8,-3]*Ph[2,3,13]*Ph[9,8,13]*Pv[4,5,14]*Pv[10,12,14]
        @tensor projcenter2[-1,-2,-3,-4] := cr[11,10,-4]*cr[6,4,-2]*t1[11,7,9,12]*t1[6,1,2,5]*
                            TR[8,7,-3]*TR[3,1,-1]*Ph[10,12,14]*Ph[4,5,14]*Pv[9,8,13]*Pv[2,3,13]
    elseif direction == "RU"
        @tensor projcenter1[-1,-2,-3,-4] := cl[6,4,-2]*cl[11,10,-4]*TL[3,1,-1]*TL[7,8,-3]*t1[2,5,6,1]*
                            t1[9,12,11,7]*Ph[4,5,13]*Ph[10,12,13]*Pv[2,3,14]*Pv[9,8,14]
        @tensor projcenter2[-1,-2,-3,-4] := cr[10,11,-4]*cr[4,6,-2]*TR[7,8,-3]*TR[1,3,-1]*Pv[4,5,13]*
                            Pv[10,12,13]*Ph[9,8,14]*Ph[2,3,14]*t2[7,11,12,9]*t2[1,6,5,2]
    end
    PcL,PcR = ConstructProjectorCenter2(projcenter1,projcenter2,chimax)
    =#

    return PcL,PcR,Ph,Pv
end
