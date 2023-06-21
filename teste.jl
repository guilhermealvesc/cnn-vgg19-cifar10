using LinearAlgebra, Statistics, Flux, MLDatasets, CUDA, Plots
using BetaML: ConfusionMatrix, fit!, info
using Printf, BSON

BSON.@load "trained_model_cifar10_84.bson" model
model = gpu(model)

x_teste, y_teste = MLDatasets.CIFAR10(Float32, split=:test)[:] |> gpu

média_x_teste = mean(x_teste);
desvio_x_teste = std(x_teste);
x_teste = (x_teste .- média_x_teste) ./ desvio_x_teste;

y_teste = Flux.onehotbatch(y_teste, 0:9)

ŷteste = model(x_teste)

cm = ConfusionMatrix()
fit!(cm, Flux.onecold(y_teste) .- 1, Flux.onecold(ŷteste) .- 1)
print(cm)

info(cm)


class_names = MLDatasets.CIFAR10().metadata["class_names"]
image(x) = colorview(RGB, permutedims(x, (3, 2, 1)))

print(class_names[Flux.onecold(y_teste[:, 365])]) 
plot(image(x_teste[:, :, :, 365]))