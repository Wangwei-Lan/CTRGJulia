function NewRG(Param::Parameter1;method="NewRG")
    T,Tsx = ConstructBaseTensorDouble(Param.Beta)
    FEttrg = Array{Float64}(undef,0)
    for j in 1:Param.step
        println("------------------Step $j---------------------")
        @time begin
        #TODO  first just use RGTensor = Param.RGTensor if it's not working , then we need to modify
        #RGTensor = Param.RGTensor; RGnorm = Param.RGnorm
        #   TL = Param.TL
        #   TR = Param.TR
        #   T = RGTensor[1]
        #   Tsx = RGTensor[6]
        @time PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="LD")
        append!(Param.Isometryv,[Pv])
        append!(Param.Isometryh,[Ph])
        @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][4,6,8]*PcL[7,8,-3]*Ph[1,2,-2]*Pv[4,5,-1]*
                                        Param.RGTensor[5][5,1,3,6]*Param.TL[3,2,7]
        @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][5,4,8]*PcR[7,8,-3]*Ph[4,6,-2]*Pv[1,2,-1]*
                                        Param.RGTensor[4][5,3,1,6]*Param.TR[2,3,7]
        @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][-1,4,1,2]*Param.RGTensor[1][1,5,-3,3]*Ph[4,5,-2]*Ph[2,3,-4]
        @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #if j == 1
        #    @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[6][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #else
        #@time @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][3,1,5,-4]*Param.RGTensor[1][2,-2,4,1]*Pv[3,2,-1]*Pv[5,4,-3]
        #end
        NormalizeTensor(Param.RGTensor,Param.RGnorm)

        if method == "NewRG"
            @time PcL,PcR,Ph,Pv = Mover(Param.RGTensor,Param.TL,Param.TR,Param.chi,direction="RU")
            append!(Param.Isometryv,[Pv])
            append!(Param.Isometryh,[Ph])
            @tensor Param.RGTensor[2][-1,-2,-3] := Param.RGTensor[2][6,4,7]*Param.TL[3,2,8]*Param.RGTensor[4][1,5,6,2]*
                                        Ph[4,5,-2]*Pv[1,3,-1]*PcL[8,7,-3]
            @tensor Param.RGTensor[3][-1,-2,-3] := Param.RGTensor[3][4,5,7]*Param.TR[2,3,8]*Param.RGTensor[5][2,5,6,1]*
                                        Ph[1,3,-2]*Pv[4,6,-1]*PcR[8,7,-3]
            @tensor Param.RGTensor[5][-1,-2,-3,-4] := Param.RGTensor[5][1,5,-3,3]*Param.RGTensor[1][-1,4,1,2]*Ph[5,4,-2]*Ph[3,2,-4]
            @tensor Param.RGTensor[4][-1,-2,-3,-4] := Param.RGTensor[4][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
            #@time @tensor Param.RGTensor[6][-1,-2,-3,-4] := Param.RGTensor[6][2,-2,4,1]*Param.RGTensor[1][3,1,5,-4]*Pv[2,3,-1]*Pv[4,5,-3]
            NormalizeTensor(Param.RGTensor,Param.RGnorm)
        end
        end

        #
        #Ns = (2*(j+1))^2
        #=
        Ns = 1+8*sum(1:1:j)
        #Ns = (2*(j+1))^2
        @tensor Z[] := Param.RGTensor[2][1,2,3]*Param.RGTensor[3][1,2,3]
        #println(Z[1])
        FE = (log(Z[1]) + sum(log.(Param.RGnorm[2]))+ sum(log.(Param.RGnorm[3]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j-1:-1:0...].*log.(Param.RGnorm[4][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[4][2:2:end]))+
                sum([j:-1:1...].*log.(Param.RGnorm[5][1:2:end]))+sum([j-1:-1:0...].*log.(Param.RGnorm[5][2:2:end])))/Ns
        append!(Param.FEttrg,[FE])
        =#
        #=
        @tensor Numerator[] := Tsx[17,3,5,1]*T[19,2,7,3]*Tsx[15,11,17,9]*T[16,10,19,11]*Param.RGTensor[4][18,1,4,2]*
                        Param.RGTensor[4][12,9,18,10]*Param.RGTensor[5][5,8,15,6]*Param.RGTensor[5][7,14,16,8]*
                        Param.RGTensor[2][4,6,13]*Param.RGTensor[3][12,14,13]
        @tensor Denominator[] := T[17,3,5,1]*T[19,2,7,3]*T[15,11,17,9]*T[16,10,19,11]*Param.RGTensor[4][18,1,4,2]*
                        Param.RGTensor[4][12,9,18,10]*Param.RGTensor[5][5,8,15,6]*Param.RGTensor[5][7,14,16,8]*
                        Param.RGTensor[2][4,6,13]*Param.RGTensor[3][12,14,13]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #=
        @tensor Numerator[] := Tsx[18,3,4,1]*Tsx[4,6,10,7]*T[21,2,5,3]*T[5,9,13,6]*T[17,20,18,23]*T[19,24,21,20]*
                        Param.RGTensor[4][25,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][22,23,25,24]*
                        Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,19,14]*Param.RGTensor[2][12,11,15]*
                        Param.RGTensor[3][22,16,15]
        @tensor Denominator[] := T[18,3,4,1]*T[4,6,10,7]*T[21,2,5,3]*T[5,9,13,6]*T[17,20,18,23]*T[19,24,21,20]*
                        Param.RGTensor[4][25,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][22,23,25,24]*
                        Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,19,14]*Param.RGTensor[2][12,11,15]*
                        Param.RGTensor[3][22,16,15]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #=
        @tensor Numerator[] := Tsx[24,3,4,1]*Tsx[4,6,10,7]*T[27,2,5,3]*T[17,19,23,20]*T[23,26,24,29]*T[18,22,25,19]*T[25,31,27,26]*
                                T[5,9,13,6]*Param.RGTensor[4][30,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][21,20,28,22]*
                                Param.RGTensor[4][28,29,30,31]*Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,18,14]*
                                Param.RGTensor[2][12,11,15]*Param.RGTensor[3][21,16,15]
        @tensor Denominator[] := T[24,3,4,1]*T[4,6,10,7]*T[27,2,5,3]*T[17,19,23,20]*T[23,26,24,29]*T[18,22,25,19]*T[25,31,27,26]*
                                T[5,9,13,6]*Param.RGTensor[4][30,1,8,2]*Param.RGTensor[4][8,7,12,9]*Param.RGTensor[4][21,20,28,22]*
                                Param.RGTensor[4][28,29,30,31]*Param.RGTensor[5][10,14,17,11]*Param.RGTensor[5][13,16,18,14]*
                                Param.RGTensor[2][12,11,15]*Param.RGTensor[3][21,16,15]
        internalenergytree = Numerator[1]/Denominator[1]
        append!(Param.IEttrg,[internalenergytree])
        =#
        #=
        Ns = 2*(2*j+1)
        FE = FreeEnergyDensity(Param.RGTensor,Param.RGnorm,j)
        append!(Param.FEttrg,[-1*FE/(Param.Beta)/2])
        =#
    end
end
