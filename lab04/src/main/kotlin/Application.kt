import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.io.File
import java.util.*

object Application {
    private val log = LoggerFactory.getLogger(Application::class.java)

    private const val INPUT_WIDTH = 28
    private const val INPUT_HEIGHT = 28
    private const val INPUT_NUM_CHANNELS = 1

    private const val NUM_CLASSES = 10

    private const val NUM_EPOCHS = 1

    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty()) {
            log.error("Datasets path required!")
            return
        }

        val basePath = args[0]
        val trIterator = prepareDataset("$basePath/training")
        val tsIterator = prepareDataset("$basePath/testing")

        val config = configLeNet()
        val model = prepareModel(config)

        log.info("Number of trained parameters: {}", model.numParams())

        for (i in 0 until NUM_EPOCHS) {
            log.info("---------- EPOCH {} ----------", i + 1)
            log.info("Training...")

            model.fit(trIterator)

            log.info("Testing...")

            val eval = model.evaluate<Evaluation>(tsIterator)

            log.info(eval.stats())

            trIterator.reset()
            tsIterator.reset()

            log.info("---------- EPOCH COMPLETE ----------")
        }
    }

    private fun prepareDataset(path: String): RecordReaderDataSetIterator {
        val batchSize = 64
        val randSeed = 1234

        val random = Random(randSeed.toLong())

        val data = File(path)
        val split = FileSplit(data, NativeImageLoader.ALLOWED_FORMATS, random)

        val labelMaker = ParentPathLabelGenerator()
        val reader = ImageRecordReader(INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong(), INPUT_NUM_CHANNELS.toLong(), labelMaker)
        reader.initialize(split)

        val iterator = RecordReaderDataSetIterator(reader, batchSize, 1, NUM_CLASSES)

        val scaler = ImagePreProcessingScaler(0.0, 1.0)
        scaler.fit(iterator)
        iterator.preProcessor = scaler

        return iterator
    }

    private fun prepareModel(conf: MultiLayerConfiguration): MultiLayerNetwork {
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(10))
        return net
    }

    private fun configLeNet(): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
                .updater(Adam())
                .weightInit(WeightInit.NORMAL)
                .list()
                .layer(0, ConvolutionLayer.Builder()
                        .nIn(INPUT_NUM_CHANNELS)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, ConvolutionLayer.Builder()
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong(), INPUT_NUM_CHANNELS.toLong()))
                .build()
    }
}
