# ENV["JULIA_CUDA_SILENT"] = true
using LinearAlgebra, Statistics, Flux, MLDatasets, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf
using Images

# Treino
class_names = MLDatasets.CIFAR10().metadata["class_names"]
x_treino, y_treino = MLDatasets.CIFAR10(split=:train)[:]
x_treino = permutedims(x_treino, (2, 1, 3, 4)); # For correct img axis
x_treino = convert(Array{Float32, 4}, x_treino);
x_treino = reshape(x_treino, (32, 32, 3, 50000));
y_treino = Flux.onehotbatch(y_treino, 0:9)
dados_treino = Flux.Data.DataLoader((x_treino, y_treino), batchsize=128)

média_x_treino    = mean(x_treino);
desvio_x_treino   = std(x_treino);
x_treino          = (x_treino .- média_x_treino) ./ desvio_x_treino;

# Teste
x_teste, y_teste = MLDatasets.CIFAR10(split=:test)[:]
x_teste = permutedims(x_teste, (2, 1, 3, 4)); # For correct img axis
x_teste = convert(Array{Float32, 4}, x_teste);
x_teste = reshape(x_teste, (32, 32, 3, 10000));

média_x_teste     = mean(x_teste);
desvio_x_teste    = std(x_teste);
x_teste           = (x_teste .- média_x_teste) ./ desvio_x_teste;

y_teste = Flux.onehotbatch(y_teste, 0:9)

modelo = Chain(
   Conv((5,5), 3=>16, relu),
   MaxPool((2,2)),
   Conv((5,5), 16=>8, relu),
   MaxPool((2,2)),
   x -> reshape(x, :, size(x, 4)),
   Dense(200, 120),
   Dense(120, 84),
   Dense(84, 10),
   Flux.softmax
)

acuracia(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
perda(x, y) = Flux.crossentropy(modelo(x), y)

opt = Flux.ADAM()
ps = Flux.params(modelo)

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

ŷtreino = modelo(x_treino)
ŷteste = modelo(x_teste)

acuracia(ŷtreino, y_treino)
acuracia(ŷteste, y_teste)

image(x) = colorview(RGB, permutedims(x, (3, 1, 2)))

class_names[Flux.onecold(y_treino[:, 150])]  # rótulo da amostra 2
plot(image(x_treino[:, :, :, 150]))

cm = ConfusionMatrix()
fit!(cm, Flux.onecold(y_teste) .- 1, Flux.onecold(ŷteste) .- 1)
print(cm)

res = info(cm)

heatmap(string.(res["categories"]),
   string.(res["categories"]),
   res["normalised_scores"],
   seriescolor=cgrad([:white, :blue]),
   xlabel="Predito",
   ylabel="Real",
   title="Matriz de Confusão (scores normalizados)")

# Limita o mapa de cores, para vermos melhor onde os erros estão

heatmap(string.(res["categories"]),
   string.(res["categories"]),
   res["normalised_scores"],
   seriescolor=cgrad([:white, :blue]),
   clim=(0.0, 0.02),
   xlabel="Predito",
   ylabel="Real",
   title="Matriz de Confusão (scores normalizados)")