
import numpy as np
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D, TimeDistributed, GlobalMaxPooling1D, Dense, Reshape,Concatenate, Multiply, Permute, Conv1D
from tensorflow.keras.layers import Add, Activation, Lambda

# define channel attention
def channel_attention(inputs, ratio=8):
    channel_axis = -1
    time_points = inputs.shape[1]
    # get channel
    channel = int(inputs.shape[channel_axis])
    Dense_one = TimeDistributed(Dense(channel//ratio, activation='elu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros'))
    Dense_two = TimeDistributed(Dense(channel, kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros'))
    # average pooling
    avg_pool = TimeDistributed(GlobalAveragePooling1D())(inputs)
    avg_pool = TimeDistributed(Reshape((1,channel)))(avg_pool)
    assert avg_pool.shape[1:] == (time_points,1,channel)
    mlp_avg_pool = Dense_one(avg_pool)
    assert mlp_avg_pool.shape[1:] == (time_points,1,channel//ratio)
    mlp_avg_pool =  Dense_two(mlp_avg_pool)
    assert mlp_avg_pool.shape[1:] == (time_points,1,channel)
    # maxpooling
    max_pool = TimeDistributed(GlobalMaxPooling1D())(inputs)
    max_pool = TimeDistributed(Reshape((1,channel)))(max_pool)
    assert max_pool.shape[1:] == (time_points,1,channel)
    mlp_max_pool = Dense_one(max_pool)
    assert mlp_max_pool.shape[1:] == (time_points,1,channel//ratio)
    mlp_max_pool = Dense_two(mlp_max_pool)
    assert max_pool.shape[1:] == (time_points,1,channel)
    # add the two 
    channel_attention_feature = Add()([mlp_avg_pool, mlp_max_pool])
    channel_attention = TimeDistributed(Activation('sigmoid'))(channel_attention_feature)
    channel_attention = (Multiply())([channel_attention_feature, inputs])
    return channel_attention

# define spatial attention 
def spatial_attention(channel_attention):
    spatial_avg_pool = TimeDistributed(Lambda(lambda x: K.mean(x,axis=2,keepdims=True)))(channel_attention)
    spatial_max_pool = TimeDistributed(Lambda(lambda x: K.max(x,axis=2,keepdims=True)))(channel_attention)
    spatial_attn = (Concatenate(axis=3))([spatial_avg_pool, spatial_max_pool])
    spatial_attn = TimeDistributed(Conv1D(1,(3),padding='same',activation='sigmoid', kernel_initializer='he_normal',use_bias=False))(spatial_attn)
    return spatial_attn

def CBAM(inputs):
    channel_features = channel_attention(inputs,8)
    spatial_features = spatial_attention(channel_features)
    print(channel_features.shape)
    refined_features = Multiply()([channel_features, spatial_features])
    refined_features = Add()([refined_features, inputs])
    return refined_features
    
    
