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

    if direction == "LD"
        @tensor disentanglerhorizontal[-1,-2,-3,-4] := cl[7,10,6]*cr[3,4,6]*cl[7,9,8]*cr[5,4,8]*T[15,-2,16,14]*
                            T[13,-4,16,12]*t2[11,-1,15,10]*t2[11,-3,13,9]*t1[3,14,2,1]*t1[5,12,2,1]
        @tensor disentanglervertical[-1,-2,-3,-4] := cl[4,5,8]*cr[9,7,8]*cl[4,3,6]*cr[10,7,6]*T[12,16,-4,13]*T[14,16,-2,15]*
                            t2[2,1,12,5]*t2[2,1,14,3]*t1[9,13,-3,11]*t1[10,15,-1,11]
    elseif  direction == "RU"
        @tensor disentanglervertical[-1,-2,-3,-4] := cl[9,7,6]*cr[4,5,6]*cl[11,7,8]*cr[4,3,8]*t1[-1,10,9,12]*
                            t2[13,5,2,1]*t1[-3,10,11,14]*t2[15,3,2,1]*T[-2,12,13,16]*T[-4,14,15,16]
        @tensor disentanglerhorizontal[-1,-2,-3,-4] := cl[3,4,8]*cr[7,11,8]*cl[5,4,6]*cr[7,9,6]*t1[1,2,3,14]*t2[15,11,10,-3]*
                            t1[1,2,5,13]*t2[12,9,10,-1]*T[16,14,15,-4]*T[16,13,12,-2]
    end

    disentanglervertical = reshape(disentanglervertical,dlink*chicl1,dlink*chicl1)
    disentanglerhorizontal = reshape(disentanglerhorizontal,dlink*chicl1,dlink*chicl1)
    Fh = svd((disentanglerhorizontal+disentanglerhorizontal')/2)
    Fv = svd((disentanglervertical+disentanglervertical')/2)

    #disentanglerh = Fh.V*Fh.U
    #disentanglerv =





    #=
    projcenter1 = reshape(projcenter1,dlink^2*chicl3,dlink^2*chicl3)
    eigvalue1,eigvector1= eigenorder((projcenter1+projcenter1')/2)
    projcenter2 = reshape(projcenter2,dlink^2*chicl3,dlink^2*chicl3)
    eigvalue2,eigvector2= eigenorder((projcenter2+projcenter2')/2)
    if dlink^2*chicl3 > chimax
        PcL = reshape(eigvector1[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(eigvector2[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(eigvector1,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(eigvector2,dlink^2,chicl3,dlink^2*chicl3)
    end
    @tensor mix[-1,-2] := PcL[1,2,-1]*PcR[1,2,-2]
    F3 = svd(mix)
    @tensor PcL[-1,-2,-3] := PcL[-1,-2,1]*F3.U[1,2]*sqrt(inv(Matrix(Diagonal(F3.S))))[2,-3]
    @tensor PcR[-1,-2,-3] := sqrt(inv(Matrix(Diagonal(F3.S))))[-3,1]*F3.V[2,1]*PcR[-1,-2,2]

    projhorizontal = reshape(projhorizontal,chicl1*dlink,chicl1*dlink)
    #F = svd((projhorizontal+projhorizontal')/2)
    eigvalue,eigvector = eigenorder((projhorizontal+projhorizontal')/2)
    dlink*chicl1 > chimax ? Ph = reshape(eigvector[:,1:chimax],chicl1,dlink,chimax) :
                        Ph = reshape(eigvector,chicl1,dlink,dlink*chicl1)
    projvertical = reshape(projvertical,chicl1*dlink,chicl1*dlink)
    #F = svd((projvertical+projvertical')/2)
    eigvalue,eigvector = eigenorder((projvertical+projvertical')/2)
    dlink*chicl1 > chimax ? Pv = reshape(eigvector[:,1:chimax],chicl1,dlink,chimax) :
                            Pv= reshape(eigvector,chicl1,dlink,dlink*chicl1)
    =#
    #
    projcenter1 = reshape(projcenter1,dlink^2*chicl3,dlink^2*chicl3)
    F1 = svd((projcenter1+projcenter1')/2)
    projcenter2 = reshape(projcenter2,dlink^2*chicl3,dlink^2*chicl3)
    F2 = svd((projcenter2+projcenter2')/2)
    #=
    PcL = reshape(F1.U,dlink^2,chicl3,dlink^2*chicl3)
    PcR = reshape(F2.V,dlink^2,chicl3,dlink^2*chicl3)
    =#
    #
    if dlink^2*chicl3 > chimax
        PcL = reshape(F1.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F2.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F1.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F2.V,dlink^2,chicl3,dlink^2*chicl3)
    end
    #
    @tensor mix[-1,-2] := PcL[1,2,-1]*PcR[1,2,-2]
    #mix = F1.Vt[1:size(PcL,3),:]*F2.U[:,1:size(PcR,3)]
    F3 = svd(mix)
    #println(F3.S)
    #@tensor PcL[-1,-2,-3] := PcL[-1,-2,1]*F3.U[1,2]*sqrt(inv(Matrix(Diagonal(F3.S))))[2,-3]
    #@tensor PcR[-1,-2,-3] := sqrt(inv(Matrix(Diagonal(F3.S))))[-3,1]*F3.V[2,1]*PcR[-1,-2,2]
    @tensor PcL[-1,-2,-3] := PcL[-1,-2,1]*F3.U[1,2]*sqrt((Matrix(Diagonal(F3.S))))[2,-3]
    @tensor PcR[-1,-2,-3] := sqrt((Matrix(Diagonal(F3.S))))[-3,1]*F3.V[2,1]*PcR[-1,-2,2]
    #
    if dlink^2*chicl3 > chimax
        PcL = reshape(PcL[:,:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(PcR[:,:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(PcL,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(PcR,dlink^2,chicl3,dlink^2*chicl3)
    end
    #
    #=
    projcenter = reshape(projcenter,dlink^2*chicl3,dlink^2*chicl3)
    F = svd((projcenter+projcenter')/2)
    if dlink^2*chicl3 > chimax
        PcL = reshape(F.U[:,1:chimax],dlink^2,chicl3,chimax)
        PcR = reshape(F.V[:,1:chimax],dlink^2,chicl3,chimax)
    else
        PcL = reshape(F.U,dlink^2,chicl3,dlink^2*chicl3)
        PcR = reshape(F.V,dlink^2,chicl3,dlink^2*chicl3)
    end
    PcR = PcL
    =#
    projhorizontal = reshape(projhorizontal,chicl1*dlink,chicl1*dlink)
    F = svd((projhorizontal+projhorizontal')/2)
    dlink*chicl1 > chimax ? Ph = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                        Ph = reshape(F.U,chicl1,dlink,dlink*chicl1)
    projvertical = reshape(projvertical,chicl1*dlink,chicl1*dlink)
    F = svd((projvertical+projvertical')/2)
    dlink*chicl1 > chimax ? Pv = reshape(F.U[:,1:chimax],chicl1,dlink,chimax) :
                            Pv= reshape(F.U,chicl1,dlink,dlink*chicl1)
    #
    return PcL,PcR,Ph,Pv
end
