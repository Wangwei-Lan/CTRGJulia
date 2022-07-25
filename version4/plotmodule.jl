

test = (-2*InternalEnergy[1].-InternalEnergyExact)./InternalEnergyExact
append!(IEtest,[abs.(test)])

test = (FreeEnergy[1].-FreeEnergyExact)./FreeEnergyExact
append!(FEtest,[abs.(test)])

x = 1


 using PyPlot

iterations = length(IEtest[1])
semilogy(1:iterations,IEtest[1],"v--",label="χ=8")
semilogy(1:iterations,IEtest[2],"v--",label="χ=10")
semilogy(1:iterations,IEtest[3],"v--",label="χ=12")
semilogy(1:iterations,IEtest[4],"v--",label="χ=14")
semilogy(1:iterations,IEtest[5],"v--",label="χ=16")
semilogy(1:iterations,IEtest[6],"v--",label="χ=18")
semilogy(1:iterations,IEtest[7],"v--",label="χ=20")
semilogy(1:iterations,IEtest[8],"v--",label="χ=22")
semilogy(1:iterations,IEtest[9],"v--",label="χ=24")
semilogy(1:iterations,IEtest[10],"v--",label="χ=26")
semilogy(1:iterations,IEtest[11],"v--",label="χ=28")
semilogy(1:iterations,IEtest[12],"v--",label="χ=30")




legend()


semilogy(1:iterations,FEtest[1],"v--",label="χ=8")
semilogy(1:iterations,FEtest[2],"v--",label="χ=10")
semilogy(1:iterations,FEtest[3],"v--",label="χ=12")
semilogy(1:iterations,FEtest[4],"v--",label="χ=14")
semilogy(1:iterations,FEtest[5],"v--",label="χ=16")
semilogy(1:iterations,FEtest[6],"v--",label="χ=18")
semilogy(1:iterations,FEtest[7],"v--",label="χ=20")
semilogy(1:iterations,FEtest[8],"v--",label="χ=22")
semilogy(1:iterations,FEtest[9],"v--",label="χ=24")
semilogy(1:iterations,FEtest[10],"v--",label="χ=26")
semilogy(1:iterations,FEtest[11],"v--",label="χ=28")
semilogy(1:iterations,FEtest[12],"v--",label="χ=30")

legend()





iterations = length(IEtest[1])
semilogy(1:2:iterations,IEtest[1][1:2:end],"o--",label="χ=8")
semilogy(1:2:iterations,IEtest[2][1:2:end],"o--",label="χ=10")
semilogy(1:2:iterations,IEtest[3][1:2:end],"o--",label="χ=12")
semilogy(1:2:iterations,IEtest[4][1:2:end],"o--",label="χ=14")
semilogy(1:2:iterations,IEtest[5][1:2:end],"o--",label="χ=16")
semilogy(1:2:iterations,IEtest[6][1:2:end],"o--",label="χ=18")
semilogy(1:2:iterations,IEtest[7][1:2:end],"o--",label="χ=20")
semilogy(1:2:iterations,IEtest[8][1:2:end],"o--",label="χ=22")
semilogy(1:2:iterations,IEtest[9][1:2:end],"o--",label="χ=24")
semilogy(1:2:iterations,IEtest[10][1:2:end],"o--",label="χ=26")
semilogy(1:2:iterations,IEtest[11][1:2:end],"o--",label="χ=28")
semilogy(1:2:iterations,IEtest[12][1:2:end],"o--",label="χ=30")




legend()


semilogy(1:2:iterations,FEtest[1][1:2:end],"v--",label="χ=8")
semilogy(1:2:iterations,FEtest[2][1:2:end],"v--",label="χ=10")
semilogy(1:2:iterations,FEtest[3][1:2:end],"v--",label="χ=12")
semilogy(1:2:iterations,FEtest[4][1:2:end],"v--",label="χ=14")
semilogy(1:2:iterations,FEtest[5][1:2:end],"v--",label="χ=16")
semilogy(1:2:iterations,FEtest[6][1:2:end],"v--",label="χ=18")
semilogy(1:2:iterations,FEtest[7][1:2:end],"v--",label="χ=20")
semilogy(1:2:iterations,FEtest[8][1:2:end],"v--",label="χ=22")
semilogy(1:2:iterations,FEtest[9][1:2:end],"v--",label="χ=24")
semilogy(1:2:iterations,FEtest[10][1:2:end],"v--",label="χ=26")
semilogy(1:2:iterations,FEtest[11][1:2:end],"v--",label="χ=28")
semilogy(1:2:iterations,FEtest[12][1:2:end],"v--",label="χ=30")

legend()



semilogy(1:iterations,FEtest[1],"bv--",label="χ=8")
semilogy(1:iterations,FEtest[2],"rv--",label="χ=9")
semilogy(1:iterations,FEtest[3],"gv--",label="χ=10")
semilogy(1:iterations,FEtest[4],"mv--",label="χ=11")
semilogy(1:iterations,FEtest[5],"cv--",label="χ=12")
semilogy(1:iterations,FEtest[6],"yv--",label="χ=13")
semilogy(1:iterations,FEtest[7],"v--",label="χ=14")
semilogy(1:iterations,FEtest[8],"v--",label="χ=15")
semilogy(1:iterations,FEtest[9],"v--",label="χ=16")


legend()


semilogy(1:500,FEtest[1],"bv--",label="χ=4")
semilogy(1:500,FEtest[2],"rv--",label="χ=8")
semilogy(1:500,FEtest[3],"gv--",label="χ=10")
semilogy(1:500,FEtest[4],"mv--",label="χ=16")
semilogy(1:500,FEtest[5],"cv--",label="χ=20")
semilogy(1:500,FEtest[6],"yv--",label="χ=25")

legend()
