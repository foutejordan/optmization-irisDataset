using CSV
using Ipopt
using JuMP
using DataFrames
#X: l'ensemble des donnees d'entrainement
#t: les labels

function svmdual_with_error(X, t)

    m = Model(Ipopt.Optimizer)

    N = size(X)[1]
    n = size(X)[2]


    M=1000

    e = [1 for i=1:N]

    Q = zeros(N, N)


    for i=1:N
        for j = 1:N
            Q[i,j] =  t[i] * t[j] * (sum(X[i,k] * X[j,k] for k=1:n) + 0.5).^2     
        end
    end
    println(Q)

 
    @variable(m, 0 <= λ[1:N] <= M)
    #sum(e[i] * λ[i] for i=1:N)
    #fonction objective
   @expression(m, expr, e' * λ  - 0.5 * λ' * Q * λ)
   
   @NLobjective(m, Max, expr)

   #@NLobjective(m, Max, sum(e[i] * λ[i] for i=1:N) - 0.5 * (sum(λ[j] * Q[i][j]*λ[j] for j=1:N)) )

    #contraintes
    #
    @constraint(m, sum(λ[j]*t[j] for j=1:N) == 0 )

    status = optimize!(m)
    objective_value(m)

    lamda = value.(λ)
    # calcul de b
    S = length(λ)
    println(S)
    b = 1/S * sum(t[i] - sum(lamda[j] * t[i] * (sum(X[i,k] * X[j,k] for k=1:n) + 0.5).^2  for j=1:N) for i=1:N)
    
    println(b)
    return b,value.(λ)
end

# Ouverture du fichier "iris.csv"
using DelimitedFiles
#question 1

function init(filename,delim)
	res = readdlm(filename,delim)
	N = size(res)[1]
	n = size(res)[2]
    println(N)
    println(n)

	X = zeros(N-1, n-2)
	T  = zeros(N-1)

	for i = 2:N
		for j = 2:n-1
			X[i-1,j-1] = res[i,j]
		end
        if (res[i,n] == "Iris-setosa") 
            T[i-1] = 1
        elseif (res[i,n] == "Iris-versicolor")
            T[i-1] = 2
        else 
            T[i-1] = 3
        end
	end		
	return([X,T])
end

data = init("shuffled_Iris.csv", ',')

println(data)

svmdual_with_error(data[1], data[2])

#fonction de calcul du pourcentage d'ambiguite

function percent(testdata)
    x = testdata[1]
    t = testdata[2]

    N = size(x)[1]
    n = size(x)[2]
    y1 = zeros(N)
    y2 = zeros(N)
    p = 0

    for i=1:N
        y1[i] = 0
        y2[i] = 0
        for j=1:N 
            y1[i] += λ[i] * t[i] *  (sum(x[i,k] * x[j,k] for k=1:n) + 0.5).^2

            y2[i] += λ[i] * t[i] *  (sum(x[i,k] * x[j,k] for k=1:n) + 0.5).^2
        end
        y1[i] += b
        y2[i] += b
        if(y1[i] > 0 && y2[i] > 0)
            p = p + 1
            
        end
    end
    percent = (p / N) * 100
    println(percent)
end



