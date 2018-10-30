from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.core.urlresolvers import reverse
from django.views.generic import TemplateView
from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from user.models import *
from dic.models import *
from datetime import datetime, date, timezone
from ipware.ip import get_ip
import html, difflib, os
from django.core.mail import send_mail
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers import core as layers_core
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

def interpolate(v1, v2, step):
	ret = []
	for i in range(1, step + 1):
		ret.append([v1[j] + (v2[j] - v1[j]) * i / step for j in range(len(v1))])
	return ret

def sent_ig(sent, star):
	### constants
	UNKNOWN = 0
	GO = 1
	END = 2
	PAD = 3

	### setting arguments
	word2vec = {}
	word2idx = {}
	idx2word = {UNKNOWN:'?', 1:'_1_', 2:'_2_', 3:'_3_', 4:'_4_', 5:'_5_', END:''}
	word2vec_dim = 0
	word_list = []
	vocab_dim = 4
	latent_dim = 50
	max_sent_len = 20
	learning_rate = 0.003
	use_word2vec = False
	vismode = 'ig'

	with open('data/word_list.data', encoding='utf-8') as fp:
		words = fp.read().strip().split('\n')
		for word in words:
			word_list.append(word)
			word2idx[word] = vocab_dim
			idx2word[vocab_dim] = word
			vocab_dim += 1
	embedding_dim = 100

	test_x = []
	test_x_len = []	
	test_y = []
	test_mask = []
	test_star = []
	test_size = 0

	### read data
	sent_idx = []
	sent_mask = []
	words = sent.split()
	star = star
	for word in words:
		if word in word2idx:
			sent_idx.append(word2idx[word])
			sent_mask.append(1.0)
		else:
			sent_idx.append(UNKNOWN)
			sent_mask.append(0.0)
		if len(sent_idx) >= max_sent_len:
			break
	test_x.append([GO] + sent_idx + [PAD for _ in range(max_sent_len - len(sent_idx))])
	test_y.append(sent_idx + [END] + [PAD for _ in range(max_sent_len - len(sent_idx))])
	test_mask.append(sent_mask + [1.0] + [0.0 for _ in range(max_sent_len - len(sent_idx))])
	test_x_len.append(max_sent_len + 1)
	test_star.append([1 if i + 1 == star else 0 for i in range(5)])
	test_size += 1

	max_sent_len += 1 # +1 for _star_ and <END>

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.int32, [None, max_sent_len])
	X_len = tf.placeholder(tf.int32, [None])
	Y = tf.placeholder(tf.int32, [None, max_sent_len])
	Y_len = tf.placeholder(tf.int32, [None])
	Y_mask = tf.placeholder(tf.float32, [None, max_sent_len])
	Star = tf.placeholder(tf.float32, [None, 5])

	inputs_enc = layers.embed_sequence(X, vocab_size=vocab_dim, embed_dim=embedding_dim)
	outputs_enc = layers.embed_sequence(Y, vocab_size=vocab_dim, embed_dim=embedding_dim)
	cell_enc = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, inputs=inputs_enc, sequence_length=X_len, dtype=tf.float32, scope='g1')
	cell_dec = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim // 2, state_is_tuple=False)
	helper_train = tf.contrib.seq2seq.TrainingHelper(outputs_enc, Y_len)
	g1 = tf.concat([state_enc.h, Star], axis=-1)
	latent = tf.layers.dense(g1, latent_dim)
	init = latent
	projection_layer = layers_core.Dense(vocab_dim, use_bias=False)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_dec, helper=helper_train, initial_state=init, output_layer=projection_layer)
	outputs_dec, last_state, last_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, impute_finished=True, maximum_iterations=max_sent_len)
	loss = tf.contrib.seq2seq.sequence_loss(logits=outputs_dec.rnn_output, targets=Y, weights=Y_mask)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, 'data/model')

		# feed_dict={X:data_x, X_len:data_x_len, Y:data_y, Y_len:data_x_len, Y_mask:data_mask, Star:data_star}
		# ret = sess.run(outputs_dec, feed_dict=feed_dict)

		# sent_gen = [idx2word[idx] for idx in ret.sample_id[0]]

		### get IG info
		# t_grads = tf.gradients(outputs_dec, inputs_enc)
		# grads, ie = sess.run([t_grads, inputs_enc], feed_dict={X:interpolate([0 for i in range(max_sent_len)], data_x[0], 100), X_len:[data_x_len[0]] * 100, Y:[data_y[0]] * 100, Y_len:[data_x_len[0]] * 100, Y_mask:[data_mask[0]] * 100, Star:[data_star[0]] * 100})
		# grads = np.array(grads)
		# ie = np.array(ie[0]) # select [0] since we calc 100 data for interpolation
		# agrads = np.average(grads, axis=1)[0]
		# ig = []
		# for i in range(max_sent_len):
		# 	t = 0.0
		# 	for j in range(embedding_dim):
		# 		t += ie[i][j] * agrads[i][j]
		# 	ig.append(t)

		ig_list = []
		sent_gen_list = []
		node_ig1_list = []
		node_ig2_list = []
		node_ig3_list = []

		### get tensor values
		t_grads = tf.gradients(outputs_dec, inputs_enc)
		t_grads1 = tf.gradients(g1, inputs_enc)
		# t_grads2 = tf.gradients(outputs_dec, g2)
		t_grads3 = tf.gradients(latent, inputs_enc)
		ret, grads, grads1, grads3, ie, _g1, _g3 = sess.run([outputs_dec, t_grads, t_grads1, t_grads3, inputs_enc, g1, latent], feed_dict={X:interpolate([0 for i in range(max_sent_len)], test_x[0], 100), X_len:[test_x_len[0]] * 100, Y:[test_y[0]] * 100, Y_len:[test_x_len[0]] * 100, Y_mask:[test_mask[0]] * 100, Star:[test_star[0]] * 100})

		sent_gen = ' '.join([idx2word[idx] for idx in ret.sample_id[0]])

		### get IG info
		grads = np.array(grads)
		ie = np.array(ie[0]) # select [0] since we calc 100 data for interpolation
		agrads = np.average(grads, axis=1)[0]
		ig = []
		for i in range(max_sent_len):
			t = 0.0
			for j in range(embedding_dim):
				t += ie[i][j] * agrads[i][j]
			ig.append(t)

		ig_list.append(ig)
		sent_gen_list.append(sent_gen)

		#################### get IG of g1/g2/latent ####################
		### get IG of g1
		grads1 = np.array(grads1)
		_g1 = np.array(_g1[0]) # select [0] since we calc 100 data for interpolation
		agrads = np.average(grads1, axis=1)[0]

		ig = []
		for i in range(max_sent_len):
			t = 0.0
			for j in range(embedding_dim):
				t += ie[i][j] * agrads[i][j]
			ig.append(t)

		node_ig1_list.append(ig)

		### get IG of g2
		# grads2 = np.array(grads2)
		# _g2 = np.array(_g2[0]) # select [0] since we calc 100 data for interpolation
		# agrads = np.average(grads2, axis=1)[0]

		# ig = []
		# for i in range(latent_dim * 2):
		# 	t = _g2[i] * agrads[i]
		# 	ig.append(t)

		# node_ig2_list.append(ig)

		### get IG of latent
		grads3 = np.array(grads3)
		_g3 = np.array(_g3[0]) # select [0] since we calc 100 data for interpolation
		agrads = np.average(grads3, axis=1)[0]

		ig = []
		for i in range(max_sent_len):
			t = 0.0
			for j in range(embedding_dim):
				t += ie[i][j] * agrads[i][j]
			ig.append(t)

		node_ig3_list.append(ig)


	return sent_gen_list[0], ig_list[0], node_ig1_list[0], node_ig3_list[0]

def softmax(vec):
	vec = [v - max(vec) for v in vec]
	ex = [2.71828 ** v for v in vec]
	exs = sum(ex)
	return [e / exs for e in ex]

def rescale(vec, full=1, base='zero'):
	min_val = min(vec)
	max_val = max(vec)
	if base == 'min':
		base = min_val
	elif base == 'max':
		base = max_val
	elif base == 'zero':
		base = 0
	return [full * (v - base) / (max_val - min_val) for v in vec]

def view_index(request):
	userinfo = None
	if request.user.is_authenticated():
		userinfo = UserInfo.objects.get(user=request.user)
	show_ig = False
	sent = ''
	star = 3
	ig_list = []
	ig_word_pair = []
	ig_word_pair_detail = []
	err = False
	err_msg = ''
	sent_gen = []
	sent_origin = ''
	if request.method == 'POST':
		sent_origin = request.POST['sent'].strip()
		sent = word_tokenize(sent_origin.lower())
		star = int(request.POST['star'])
		if len(sent) > 12:
			err = True
			err_msg = 'You can see results up to 20 words in length.' # 최대 20단어까지만 결과를 확인하실 수 있습니다.
		elif len(sent) < 4:
			err = True
			err_msg = 'You must enter a minimum of four words to see the results.' # 최소 4단어 이상 입력해야 결과를 확인하실 수 있습니다.
		else:
			sent_temp, ig_list, ig_list_1, ig_list_3 = sent_ig(' '.join(sent), star)
			sent_gen = []
			for word in sent_temp:
				sent_gen.append(word)
				if word in ['.', '?', '!']: break
			ig_list = list(map(lambda x: min(abs(int(x + 0.001)), 100), rescale(ig_list[1:][:len(sent)], full=100, base='zero')))
			ig_list_1 = list(map(lambda x: min(abs(int(x + 0.001)), 100), rescale(ig_list_1, full=100, base='zero')))
			ig_list_3 = list(map(lambda x: min(abs(int(x + 0.001)), 100), rescale(ig_list_3, full=100, base='zero')))
			ig_word_pair = [{'ig':ig_list[i], 'ig_rev':max(100 - ig_list[i], 0), 'sent':sent[i]} for i in range(len(sent))]
			ig_word_pair_detail = [{'ig1':ig_list_1[i], 'ig1_rev':max(100 - ig_list_1[i], 0), 'ig3':ig_list_3[i], 'ig3_rev':max(100 - ig_list_3[i], 0), 'out':ig_list[i], 'out_rev':max(100 - ig_list[i], 0), 'sent':sent[i]} for i in range(len(sent))]
			show_ig = True
	data = {
		'userinfo':userinfo,
		'sent':sent,
		'star':star,
		'star_range':range(star),
		'sent_gen':sent_gen,
		'sent_origin':sent_origin,
		'ig_list':ig_list,
		'ig_word_pair':ig_word_pair,
		'ig_word_pair_detail':ig_word_pair_detail,
		'show_ig':show_ig,
		'err':err,
		'err_msg':err_msg
	}
	return render(request, 'ai/index.html', data)