package org.deeplearning4j.nn.modelimport.keras.layers.core;

import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import java.util.Map;

public class KerasUnstack extends KerasLayer {

        private ElementWiseVertex.Op mergeMode = null;

        /**
         * Pass-through constructor from KerasLayer
         *
         * @param kerasVersion major keras version
         * @throws UnsupportedKerasConfigurationException Unsupported Keras config
         */
        public KerasUnstack(Integer kerasVersion) throws UnsupportedKerasConfigurationException {
            super(kerasVersion);
        }

        /**
         * Constructor from parsed Keras layer configuration dictionary.
         *
         * @param layerConfig dictionary containing Keras layer configuration.
         * @throws InvalidKerasConfigurationException     Invalid Keras config
         * @throws UnsupportedKerasConfigurationException Unsupported Keras config
         */
        public KerasUnstack(Map<String, Object> layerConfig, int index, int totalstacksize)
                throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
            this(layerConfig, true, index, totalstacksize);
        }

        /**
         * Constructor from parsed Keras layer configuration dictionary and merge mode passed in.
         *
         * @param layerConfig           dictionary containing Keras layer configuration
         * @param enforceTrainingConfig whether to enforce training-related configuration options
         * @throws InvalidKerasConfigurationException     Invalid Keras config
         * @throws UnsupportedKerasConfigurationException Unsupported Keras config
         */
        public KerasUnstack(Map<String, Object> layerConfig, boolean enforceTrainingConfig, int index, int totalstacksize)
                throws InvalidKerasConfigurationException, UnsupportedKerasConfigurationException {
                super(layerConfig, enforceTrainingConfig);
                this.vertex = new UnstackVertex(index, totalstacksize);

        }



        /**
         * Get layer output type.
         *
         * @param inputType Array of InputTypes
         * @return output type as InputType
         */
        @Override
        public InputType getOutputType(InputType... inputType) {
            return this.vertex.getOutputType(-1, inputType);
        }

}
