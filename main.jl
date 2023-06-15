ENV["JULIA_CUDA_SILENT"] = true
using LinearAlgebra, Statistics, Flux, MLDatasets, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf

x_treino, y_treino = MLDatasets.MNIST(split=:train)[:]
x_treino          = permutedims(x_treino,(2,1,3)); # For correct img axis
x_treino          = convert(Array{Float32,3},x_treino);
x_treino          = reshape(x_treino,(28,28,1,60000));
y_treino          = Flux.onehotbatch(y_treino, 0:9)
dados_treino       = Flux.Data.DataLoader((x_treino, y_treino), batchsize=128)

x_teste, y_teste  = MLDatasets.MNIST(split=:test)[:]
x_teste           = permutedims(x_teste,(2,1,3)); # For correct img axis
x_teste           = convert(Array{Float32,3},x_teste);
x_teste           = reshape(x_teste,(28,28,1,10000));
y_teste           = Flux.onehotbatch(y_teste, 0:9)

modelo = Chain(
       # 28x28 => 14x14
       Conv((5, 5), 1=>8,   pad=2, stride=2, relu),
       # 14x14 => 7x7
       Conv((3, 3), 8=>16,  pad=1, stride=2, relu),
       # 7x7 => 4x4
       Conv((3, 3), 16=>32, pad=1, stride=2, relu),
       # 4x4 => 2x2
       Conv((3, 3), 32=>32, pad=1, stride=2, relu),
       # Average pooling on each width x height feature map
       GlobalMeanPool(),
       Flux.flatten,
       Dense(32, 10),
       Flux.softmax )

acuracia(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
perda(x, y)     = Flux.crossentropy(modelo(x), y)

opt = Flux.ADAM()
ps  = Flux.params(modelo)

num_épocas = 11

for época in 1:num_épocas
  println("Época ", época)
  Flux.train!(perda, ps, dados_treino, opt)
  
  # Calcule a acurácia:
  ŷteste = modelo(x_teste)
  acu = acuracia(ŷteste, y_teste)
  
  @info(@sprintf("[%d]: Acurácia nos testes: %.4f", época, acu))
  # Se a acurácia for muito boa, termine o treino
  if acu >= 0.999
     @info(" -> Término prematuro: alcançamos uma acurácia de 99.9%")
     break
  end  
end

ŷtreino =   modelo(x_treino)
ŷteste  =   modelo(x_teste)

acuracia(ŷtreino, y_treino)
acuracia(ŷteste, y_teste)

Flux.onecold(y_treino[:,2]) - 1  # rótulo da amostra 2
plot(Gray.(x_treino[:,:,1,2]))   # imagem da amostra 2

cm = ConfusionMatrix()
fit!(cm, Flux.onecold(y_teste) .-1, Flux.onecold(ŷteste) .-1)
print(cm)

res = info(cm)

heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white,:blue]),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")

# Limita o mapa de cores, para vermos melhor onde os erros estão

heatmap(string.(res["categories"]),
        string.(res["categories"]),
        res["normalised_scores"],
        seriescolor=cgrad([:white,:blue]),
        clim=(0., 0.02),
        xlabel="Predito",
        ylabel="Real",
        title="Matriz de Confusão (scores normalizados)")