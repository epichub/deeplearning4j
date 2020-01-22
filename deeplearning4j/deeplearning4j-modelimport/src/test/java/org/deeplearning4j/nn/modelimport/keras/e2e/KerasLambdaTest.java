/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.modelimport.keras.e2e;

import com.google.flatbuffers.FlatBufferBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.string.NDArrayStrings;
import org.nd4j.resources.Resources;

import java.io.File;
import java.io.InputStream;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.List;


/**
 * Test importing Keras models with multiple Lamdba layers.
 *
 * @author Max Pumperla
 */
public class KerasLambdaTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    public class ExponentialLambda extends SameDiffLambdaLayer {
        @Override
        public SDVariable defineLayer(SameDiff sd, SDVariable x) { return x.mul(x); }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) { return inputType; }
    }

    public class TimesThreeLambda extends SameDiffLambdaLayer {
        @Override
        public SDVariable defineLayer(SameDiff sd, SDVariable x) { return x.mul(3); }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) { return inputType; }
    }


    @Test
    public void testSequentialLambdaLayerImport() throws Exception {
        KerasLayer.registerLambdaLayer("lambda_1", new ExponentialLambda());
        KerasLayer.registerLambdaLayer("lambda_2", new TimesThreeLambda());

        String modelPath = "modelimport/keras/examples/lambda/sequential_lambda.h5";

        try(InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.newFile("tempModel" + System.currentTimeMillis() + ".h5");
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            MultiLayerNetwork model = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                    .enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();

            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[]{10, 100});

            model.output(input);
        } finally {
            KerasLayer.clearLambdaLayers();
        }
    }

    @Test
    public void testModelLambdaLayerImport() throws Exception {
        KerasLayer.registerLambdaLayer("lambda_3", new ExponentialLambda());
        KerasLayer.registerLambdaLayer("lambda_4", new TimesThreeLambda());

        String modelPath = "modelimport/keras/examples/lambda/model_lambda.h5";

        try(InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.newFile("tempModel" + System.currentTimeMillis() + ".h5");
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            ComputationGraph model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath())
                    .enforceTrainingConfig(false).buildModel().getComputationGraph();

            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[]{10, 784});

            model.output(input);
        } finally {
            KerasLayer.clearLambdaLayers(); // Clear all lambdas, so other tests aren't affected.
        }
    }

    @Test
    public void testESNNModelLambdaLayerImport() throws Exception {
        //KerasLayer.registerLambdaLayer("lambda_3", new ExponentialLambda());
        //KerasLayer.registerLambdaLayer("lambda_4", new TimesThreeLambda());

        //String modelPath = "modelimport/keras/examples/lambda/model_lambda.h5";src/test/resources/modelimport/keras/weights
        String weightPath = "modelimport/keras/weights/esnn.h5";
        String modelPath = "modelimport/keras/configs/keras2/esnn.json";
        try(InputStream is = Resources.asStream(modelPath)){
            try(InputStream is2 = Resources.asStream(weightPath)) {
                File modelFile = testDir.newFile("tempModel" + System.currentTimeMillis() + ".json");
                Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                File weightFile = testDir.newFile("tempWeights" + System.currentTimeMillis() + ".h5");
                Files.copy(is2, weightFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                KerasModelBuilder modelbuilder = new KerasModel().modelBuilder();
                modelbuilder.modelJsonFilename(modelFile.getAbsolutePath());
                modelbuilder.weightsHdf5FilenameNoRoot(weightFile.getAbsolutePath());
                ComputationGraph model = modelbuilder.enforceTrainingConfig(false).buildModel().getComputationGraph();

                System.out.println(model.summary());
                INDArray input = Nd4j.create(new int[]{1, 4});
                INDArray input2 = Nd4j.create(new int[]{1, 4});
                INDArray[] inputs = new INDArray[2];
                input.putScalar(0,0.9);
                input.putScalar(1,0.9);
                input.putScalar(2,0.9);
                input.putScalar(3,0.9);
                input2.putScalar(0,0.1);
                input2.putScalar(1,0.1);
                input2.putScalar(2,0.1);
                input2.putScalar(3,0.1);
                inputs[0] = input; inputs[1] = input2;
                INDArray[] output = model.output(inputs);
                // a1 = [0.1,0.1,0.2,0.5]
                //        a2 =[0.5,0.5,0.5,0.7]
                //        ret = model.predict([[a1], [a2]])
                //        print(f"ret: :{ret}")
                //        assert ret[0] < 0.3
                // ret: :[array([[0.18835852]], dtype=float32), array([[0.04953867, 0.9504613 ]], dtype=float32), array([[0.11317298, 0.88682705]], dtype=float32)]
                System.out.println(output[0]);
                System.out.println(output[1]);
                System.out.println(output[2]);
                input.putScalar(0,0.5);
                input.putScalar(1,0.45);
                input.putScalar(2,0.45);
                input.putScalar(3,0.5);
                input2.putScalar(0,0.5);
                input2.putScalar(1,0.5);
                input2.putScalar(2,0.5);
                input2.putScalar(3,0.7);
                output = model.output(inputs);
                //ret2: :[array([[0.31219736]], dtype=float32), array([[0.09602216, 0.9039779 ]], dtype=float32), array([[0.10217763, 0.89782244]], dtype=float32)]
                System.out.println(output[0]);
                System.out.println(output[1]);
                System.out.println(output[2]);
            }
        } finally {
            KerasLayer.clearLambdaLayers(); // Clear all lambdas, so other tests aren't affected.
        }
    }

}
