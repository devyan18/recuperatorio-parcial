import * as tf from '@tensorflow/tfjs';
const inputValorCalulcar = document.getElementById('valorACalcular')
const contenedorResultado = document.getElementById('resultado')
const mostrarVisor = document.getElementById('mostrarVisor')
let modeloEntrenado

const calcularValoresY = (paramValoresX) => {
  const arrayResultadosY = [];
  for (let i = 0; i < paramValoresX.length; i++) {
    const y = 2 * paramValoresX[i] + 5;
    arrayResultadosY.push(y)
  }
  return arrayResultadosY;
}
const funcionLineal = async () => {
  contenedorResultado.innerHTML = 'El modelo se esta entrenando...';

  const X = [-1, 0, 1, 2, 3, 4];

  const Y = calcularValoresY(X)

  const xs = tf.tensor2d(X, [6, 1])
  const ys = tf.tensor2d(Y, [6, 1])
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
  model.compile({loss: 'meanSquaredError',optimizer: 'sgd', metrics: ['accuracy']})
  await model.fit(xs, ys, {
    epochs: 350,
    callbacks: [
      {onEpochEnd: async (epoch, logs) => {
          console.log('Epoch:' + epoch + ' Loss:' + logs.loss)
        }}
    ],
  })
  inputValorCalulcar.disabled = false;
  inputValorCalulcar.focus()
  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
}
document.addEventListener('DOMContentLoaded', () => {
  funcionLineal()
  inputValorCalulcar.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault()
      const valorACalcular = parseInt(inputValorCalulcar.value)
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([valorACalcular], [1, 1])
      )
      const result = resultado.dataSync()
      armarGrafica(valorACalcular, result[0])
      contenedorResultado.innerHTML = `El resultado aproximado para Y es de: ${result}`;
    }
  })
  mostrarVisor.addEventListener('click', () => {
    tfvis.visor().toggle()
  })
})
const armarGrafica = (x, y) => {
  const data = [
    {
      x: [x],
      y: [y],
      mode: 'markers',
    },
  ]
  const layout = {
    xaxis: { range: [Math.abs(x), x], title: 'Valores de X' },
    yaxis: { range: [Math.abs(y), y], title: 'Valores de Y' },
  }
  Plotly.newPlot('plot', data, layout)
}
