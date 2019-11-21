import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import layers, models
from matplotlib import ticker
import pandas as pd


# 학습 정보
batch_size = 32
epochs = 300
latent_dim = 256

# 문장 벡터화
input_texts = []
target_texts = []

input_characters = set()
target_characters = set()

data_dir = "ChatbotData.csv"

total = pd.read_csv(data_dir)

question = total["Q"].to_list()
answer = total["A"].to_list()


for i in range(0, len(question)):
    input_text, target_text = question[i], answer[i]

    target_text = '\t' + target_text + '\n'

    input_texts.append(input_text)
    target_texts.append(target_text)

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)

    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# 문자 -> 숫자 변환용 사전
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

# 학습에 사용할 데이터를 담을 3차원 배열
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')


# 문장을 문자 단위로 원 핫 인코딩하면서 학습용 데이터를 만듬
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# 숫자 -> 문자 변환용 사전
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def RepeatVectorLayer(rep, axis):
    return layers.Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis), rep, axis),
                         lambda x: tuple((x[0],) + x[1:axis] + (rep,) + x[axis:]))

# 인코더 생성
encoder_inputs = layers.Input(shape=(max_encoder_seq_length, num_encoder_tokens))
encoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)

# 디코더 생성.
decoder_inputs = layers.Input(shape=(max_decoder_seq_length, num_decoder_tokens))
decoder = layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder(decoder_inputs, initial_state=state_h)

# 어텐션 매커니즘.
repeat_d_layer = RepeatVectorLayer(max_encoder_seq_length, 2)
repeat_d = repeat_d_layer(decoder_outputs)

repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
repeat_e = repeat_e_layer(encoder_outputs)

concat_for_score_layer = layers.Concatenate(axis=-1)
concat_for_score = concat_for_score_layer([repeat_d, repeat_e])

dense1_t_score_layer = layers.Dense(latent_dim // 2, activation='tanh')
dense1_score_layer = layers.TimeDistributed(dense1_t_score_layer)
dense1_score = dense1_score_layer(concat_for_score)

dense2_t_score_layer = layers.Dense(1)
dense2_score_layer = layers.TimeDistributed(dense2_t_score_layer)
dense2_score = dense2_score_layer(dense1_score)
dense2_score = layers.Reshape((max_decoder_seq_length, max_encoder_seq_length))(dense2_score)

softmax_score_layer = layers.Softmax(axis=-1)
softmax_score = softmax_score_layer(dense2_score)

repeat_score_layer = RepeatVectorLayer(latent_dim, 2)
repeat_score = repeat_score_layer(softmax_score)

permute_e = layers.Permute((2, 1))(encoder_outputs)
repeat_e_layer = RepeatVectorLayer(max_decoder_seq_length, 1)
repeat_e = repeat_e_layer(permute_e)

attended_mat_layer = layers.Multiply()
attended_mat = attended_mat_layer([repeat_score, repeat_e])

context_layer = layers.Lambda(lambda x: K.sum(x, axis=-1),
                              lambda x: tuple(x[:-1]))
context = context_layer(attended_mat)
concat_context_layer = layers.Concatenate(axis=-1)
concat_context = concat_context_layer([context, decoder_outputs])
attention_dense_output_layer = layers.Dense(latent_dim, activation='tanh')
attention_output_layer = layers.TimeDistributed(attention_dense_output_layer)
attention_output = attention_output_layer(concat_context)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(attention_output)

# 모델 생성
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

choice = input("Load weights?")
if choice == 'y' or choice == 'Y':
    model.load_weights('att_seq2seq_weights.h5')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

#plot_model(model, show_shapes=True, to_file='model.png')
#display(Image(filename='model.png'))

choice = input("Train?")
if choice == 'y' or choice == 'Y':
    # 학습
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)

    model.save_weights('att_seq2seq_300.h5')

    # # 손실 그래프
    # plt.plot(history.history['loss'], 'y', label='train loss')
    # plt.plot(history.history['val_loss'], 'r', label='val loss')
    # plt.legend(loc='upper left')
    # plt.show()
    #
    # # 정확도 그래프
    # plt.plot(history.history['acc'], 'y', label='train acc')
    # plt.plot(history.history['val_acc'], 'r', label='val acc')
    # plt.legend(loc='upper left')
    # plt.show()

# 어텐션 검증
test_data_num = 0
test_max_len = 0

for i, s in enumerate(input_texts):
    if len(s) > test_max_len:
        test_max_len = len(s)
        test_data_num = i

test_enc_input = encoder_input_data[test_data_num].reshape(
    (1, max_encoder_seq_length, num_encoder_tokens))
test_dec_input = decoder_input_data[test_data_num].reshape(
    (1, max_decoder_seq_length, num_decoder_tokens))

attention_layer = softmax_score_layer
func = K.function([encoder_inputs, decoder_inputs] + [K.learning_phase()], [attention_layer.output])
score_values = func([test_enc_input, test_dec_input, 1.0])[0]
score_values = score_values.reshape((max_decoder_seq_length, max_encoder_seq_length))
score_values = score_values[:len(target_texts[test_data_num])-1, :len(input_texts[test_data_num])]

# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(score_values, interpolation='nearest')
# fig.colorbar(cax)

test_enc_names = []
for vec in test_enc_input[0]:
    sampled_token_index = np.argmax(vec)
    sampled_char = reverse_input_char_index[sampled_token_index]
    test_enc_names.append(sampled_char)

test_dec_names = []
for vec in test_dec_input[0]:
    sampled_token_index = np.argmax(vec)
    sampled_char = reverse_target_char_index[sampled_token_index]
    test_dec_names.append(sampled_char)


# 추론(테스트)
# 추론 모델 생성
#
encoder_model = models.Model(encoder_inputs, [encoder_outputs, state_h])
encoder_outputs_input = layers.Input(shape=(max_encoder_seq_length, latent_dim))
decoder_inputs = layers.Input(shape=(1, num_decoder_tokens))
decoder_state_input_h = layers.Input(shape=(latent_dim,))
decoder_outputs, decoder_h = decoder(decoder_inputs, initial_state=decoder_state_input_h)
repeat_d_layer = RepeatVectorLayer(max_encoder_seq_length, 2)
repeat_d = repeat_d_layer(decoder_outputs)
repeat_e_layer = RepeatVectorLayer(1, axis=1)
repeat_e = repeat_e_layer(encoder_outputs_input)
concat_for_score_layer = layers.Concatenate(axis=-1)
concat_for_score = concat_for_score_layer([repeat_d, repeat_e])

dense1_score_layer = layers.TimeDistributed(dense1_t_score_layer)
dense1_score = dense1_score_layer(concat_for_score)

dense2_score_layer = layers.TimeDistributed(dense2_t_score_layer)
dense2_score = dense2_score_layer(dense1_score)
dense2_score = layers.Reshape((1, max_encoder_seq_length))(dense2_score)

softmax_score_layer = layers.Softmax(axis=-1)
softmax_score = softmax_score_layer(dense2_score)

repeat_score_layer = RepeatVectorLayer(latent_dim, 2)
repeat_score = repeat_score_layer(softmax_score)
permute_e = layers.Permute((2, 1))(encoder_outputs_input)
repeat_e_layer = RepeatVectorLayer(1, axis=1)
repeat_e = repeat_e_layer(permute_e)
attended_mat_layer = layers.Multiply()
attended_mat = attended_mat_layer([repeat_score, repeat_e])
context_layer = layers.Lambda(lambda x: K.sum(x, axis=-1),
                              lambda x: tuple(x[:-1]))

context = context_layer(attended_mat)
concat_context_layer = layers.Concatenate(axis=-1)
concat_context = concat_context_layer([context, decoder_outputs])
attention_output_layer = layers.TimeDistributed(attention_dense_output_layer)
attention_output = attention_output_layer(concat_context)
decoder_att_outputs = decoder_dense(attention_output)
decoder_model = models.Model([decoder_inputs, decoder_state_input_h, encoder_outputs_input],
                             [decoder_outputs, decoder_h, decoder_att_outputs])

decoder_model.summary()
# #plot_model(decoder_model, show_shapes=True, to_file='decoder_model.png')
# #display(Image(filename='decoder_model.png'))


def decode_sequence(input_seq):
    # 입력 문장을 인코딩
    enc_outputs, states_value = encoder_model.predict(input_seq)

    # 디코더의 입력으로 쓸 단일 문자
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # 첫 입력은 시작 문자인 '\t'로 설정
    target_seq[0, 0, target_token_index['\t']] = 1.

    # 문장 생성
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # 이전의 출력, 상태를 디코더에 넣어서 새로운 출력, 상태를 얻음
        # 이전 문자와 상태로 다음 문자와 상태를 얻는다고 보면 됨.

        dec_outputs, h, output_tokens = decoder_model.predict(
            [target_seq, states_value, enc_outputs])

        # 사전을 사용해서 원 핫 인코딩 출력을 실제 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 종료 문자가 나왔거나 문장 길이가 한계를 넘으면 종료
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 디코더의 다음 입력으로 쓸 데이터 갱신
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = h

    return decoded_sentence


for seq_index in range(200):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('"{}" -> "{}"'.format(input_texts[seq_index], decoded_sentence.strip()))
