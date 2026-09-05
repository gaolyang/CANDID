# framework/AA2018.py
# Modified for: Binary CE + SupCon + Distillation + Replay
# Detail Level: High (Debug info, Shape checks, Metric logging)

import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict
import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Layer, Lambda
from candid_lib.base_classification_head import BaseClassificationHead
from candid_lib.base_input_encoding import BaseInputEncoding
from candid_lib.base_preprocessing import BasePreProcessing
from candid_lib.dataset_specification import DatasetSpecification
from candid_lib.enumerations import EvaluationDatasetSampling, CategoricalFormat
from candid_lib.flow_transformer_parameters import FlowTransformerParameters
from candid_lib.framework_component import FunctionalComponent
from candid_lib.model_input_specification import ModelInputSpecification
from sklearn.metrics import accuracy_score, f1_score, recall_score, balanced_accuracy_score, precision_score, confusion_matrix, \
    average_precision_score


# ==============================================================================
# 1. 闂佸搫顦悧鍡涘箠鎼淬垺鍙忔い蹇撶墕缁犳煡鏌ｉ弮鈧悘姘婵傚憡鐓曢柟杈剧到琚氶梺? Supervised Contrastive Loss (Enhanced)
# ==============================================================================
def supervised_contrastive_loss(z, labels, temperature=0.07):
    """
    z: (batch_size, feature_dim) - 闂傚鍋勫ú銈夊箠濮椻偓婵＄绠涘☉妯哄挤?L2 Normalized 闂備焦鐪归崝宀€鈧凹鍠楃粩鐔煎箳閹搭厼?
    labels: (batch_size,) - 缂傚倸鍊风粈浣肝涘☉銏″剹闁瑰鍋為崕鐔兼煥閺冨倹娅曢柛銊ㄥ亹?(int)
    """
    # 缂備胶铏庨崣搴ㄥ窗閺囩姵宕叉慨妯挎硾閸愨偓闂佹悶鍎弲鈺呭礉椤曗偓?int32 濠电偞鍨堕幐璇差潖鐠囩潿搴敊閼恒儱鍔呴梺鎸庢⒒閻℃棃宕濋埀顒傜磼?(N, 1)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels = tf.reshape(labels, (-1, 1))

    # 闂備胶顢婇妴鈧柡鍛箞閹儱顭ㄩ崼鐔告珫濠殿喗绻勯幐顩坋l 闂備胶鍎甸弲婵嬧€﹂崼銉ョ畺闁规儳顕埢鏃堟煛閸屾艾鈷旈柣鐔哥箞閺岋繝宕熼棃娑氫淮闂佹悶鍊栫划宀勵敊?(N, N)
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

    # 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘崇閸庡酣姊婚埀顒勫箛椤撶倣妤呮煙閼圭増褰у┑鈥虫喘閹繝鍩￠崨顔规寑?(N, N)
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature

    # 闂備浇妗ㄩ悞锕€顭垮鈧崺鈧い鎺戝暙娴滃綊鎳ｅ▎鎾村€甸悷娆忓椤ｆ煡鏌嶈閸庢悂宕熼瀣殘閳ь剙婀遍～瀣吋閸モ晝宕堕梻浣告啞閸ㄧ數鎹㈤幇鎵祦闁圭偓澹欓幒鏇犵杸婵ê鍚嬮弬鈧梻浣圭湽閸斿瞼鈧凹鍘奸妴鎺楁嚋闂堟稓绐為悷婊冪Ч閸┾偓?
    logits_max = tf.reduce_max(sim_matrix, axis=1, keepdims=True)
    logits = sim_matrix - logits_max

    batch_size = tf.shape(z)[0]
    logits_mask = tf.one_hot(tf.range(batch_size), batch_size)

    # 闂佸搫顦弲婊堟偡閳哄懎闂柣鎴ｅГ?mask 缂傚倷绀侀ˇ顖炩€﹀畡鎵虫瀺閹兼番鍨婚々鏌ユ煕閳╁喚娈㈤柛銈囧█閹嘲鈻庨幇顒傤儌闂佹悶鍊曢柊锝夊极瀹ュ閱囬柨婵嗘噺椤斿洭鎮楅崷顓х劸鐎规洦鍓熼、姘额敋閳ь剟骞婂☉銏╂晜闁告洦鍘鹃悾鎶芥⒑?
    mask = mask * (1 - logits_mask)

    # 闂佽崵濮崇欢銈囨閺囥垺?Log-Sum-Exp (闂備礁鎲＄敮鎺懳涘▎鎾村剨?
    exp_logits = tf.exp(logits) * (1 - logits_mask)
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-9)

    # 闂佽崵濮崇欢銈囨閺囥垺鍋╅悹鍥ф▕濞间即鏌ㄥ┑鍡橆棤濠㈢懓顦甸弻锟犲醇椤愩垹顫╁銈嗘处閸ｏ綁骞嗛崘顔肩妞ゆ劧闄勭亸娆撴⒑?Log-Likelihood
    # mask_sum 闂備礁鎼€氱兘宕归柆宥嗗剨闁斥晛鍟伴埢鏃€銇勮箛鎾村櫧濠㈢懓顦甸弻锟犲醇椤愩垹顫柣搴㈠搸閸ㄨ棄顕ｉ崹顐㈢窞閻忕偛澧介弳鐘测攽椤旂晫绠扮紒鏌ョ畺閹囨焼瀹ュ懎鐝樺銈嗗坊閸嬫挻銇勯弴銊ユ珝鐎殿噮鍋呯€靛ジ寮堕幋鐑嗕画
    mask_sum = tf.reduce_sum(mask, axis=1)

    # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞閹邦剝鎽曢梺褰掑亰娴滎亪宕电€ｎ喗鐓涚€广儱顦伴弫杈ㄧ節閳ь剟寮舵惔鎾寸參闂佹枼鏅涢崯顖滅不閹烘挻鍙忔繝鍨尵缁嬪鏌ｅΔ鍛喚鐎殿喚鏁婚、妤呭焵椤掑嫷鏁嬫い鏇楀亾闁轰礁绉瑰畷鎼併€傞崳鎱縚sum=0闂備焦瀵х粙鎴βㄩ埀顒傜磼鏉堛劎绠栫紒瀣槹濞碱亪骞嶉鍏兼珦闂傚倸鍊搁崯顐㈩嚕閸洦鏁嗘繛鎴欏灪閳锋棃鏌曢崼婵嗩仹?闂備焦瀵х粙鎴︻敊閸掔磩s闂佽崵濮崇粈浣姐亹閻愬灚?
    # 婵犵數鍋涢ˇ顓㈠礉瀹ュ绀堝ù鐓庣摠閺咁剚鎱ㄥΟ铏癸紞缂佺姴缍婂娲箵閹烘洖顏繝?1e-9 闂備礁鎼€氱兘宕归婊勫闁绘柨鎽滈々鏌ユ煕閳╁啰鈯曢柣婵堝枑缁绘稒寰勭€ｎ偆顦ラ梺浼欑祷閸旀垵鐣峰Δ鍛偓锕傚箳閺冨偆妲峰┑鐐舵彧缁茶棄螞濡ゅ懌鈧懓顦?mask_sum 闂備焦妞挎禍鐐哄窗閹炬枼鏋旈柟杈剧畱?闂備焦瀵х粙鎴﹀嫉椤掑嫬鍨傛繝濠傚椤╅攱銇勯幒鎴濃偓褰掑汲?tf.where 闁诲孩顔栭崰鏍箹椤愶箑鏄ユ俊銈呭暟?
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (mask_sum + 1e-9)

    loss = -mean_log_prob_pos
    loss = tf.where(mask_sum > 0, loss, tf.zeros_like(loss))

    return tf.reduce_mean(loss)


class FlowTransformer:
    retain_inmem_cache = False
    inmem_cache = None

    def __init__(self, pre_processing: BasePreProcessing,
                 input_encoding: BaseInputEncoding,
                 sequential_model: FunctionalComponent,
                 classification_head: BaseClassificationHead,
                 params: FlowTransformerParameters,
                 rs: np.random.RandomState = None):

        self.rs = np.random.RandomState() if rs is None else rs
        self.classification_head = classification_head
        self.sequential_model = sequential_model
        self.input_encoding = input_encoding
        self.pre_processing = pre_processing
        self.parameters = params
        self.dataset_specification: Optional[DatasetSpecification] = None
        self.X = None
        self.y = None
        self.training_mask = None
        self.model_input_spec: Optional[ModelInputSpecification] = None
        self.m = None

    def clear_memory(self):
        """Clear cached dataset tensors to reduce memory usage."""
        if hasattr(self, 'X'):
            del self.X
            self.X = None
        if hasattr(self, 'X_np_with_padding'):
            del self.X_np_with_padding
            self.X_np_with_padding = None
        if hasattr(self, 'y'):
            del self.y
            self.y = None
        gc.collect()

    # ---------------------------------------------------------------
    # 2. Build Model
    # ---------------------------------------------------------------
    def build_model(self, prefix: str = None):
        if prefix is None: prefix = ""
        # 闂備胶顭堢换瀣归崶顒夋晩?X 濠电偞鍨堕幑浣糕枍閿濆鐓橀柡宥庡幖缁秹鏌ら崫銉︽毄闁逞屽墻閸ㄤ即顢氬▎鎾村€绘俊顖涘姃缁舵艾鐣烽崷顓涘亾閿濆懎顣崇紒鈧径鎰厱婵炲棙眉濞撮攱?input_spec 闂佽瀛╃粙鎺楁晪濠电姭?
        if self.model_input_spec is None:
            raise Exception("Model Input Spec is missing!")

        inputs_dict = {}
        feature_embeddings = []
        emb_dim = 16
        win_size = self.parameters.window_size

        # --- Input Embedding Layer ---
        for numeric_feature in self.model_input_spec.numeric_feature_names:
            inp = Input((win_size, 1), name=f"{prefix}in_num_{numeric_feature}", dtype="float32")
            inputs_dict[numeric_feature] = inp
            x = Dense(emb_dim, name=f"{prefix}emb_num_{numeric_feature}")(inp)
            feature_embeddings.append(x)

        is_one_hot = (self.model_input_spec.categorical_format == CategoricalFormat.OneHot)
        for cat_feature, cat_levels in zip(self.model_input_spec.categorical_feature_names,
                                           self.model_input_spec.levels_per_categorical_feature):
            if is_one_hot:
                inp = Input((win_size, cat_levels), name=f"{prefix}in_cat_{cat_feature}", dtype="float32")
                inputs_dict[cat_feature] = inp
                x = Dense(emb_dim, name=f"{prefix}proj_cat_{cat_feature}")(inp)
            else:
                inp = Input((win_size, 1), name=f"{prefix}in_cat_{cat_feature}", dtype="int32")
                inputs_dict[cat_feature] = inp
                x = keras.layers.Embedding(input_dim=cat_levels + 1, output_dim=emb_dim,
                                           name=f"{prefix}emb_cat_{cat_feature}")(inp)
                x = keras.layers.Reshape((win_size, emb_dim))(x)
            feature_embeddings.append(x)

        if len(feature_embeddings) > 1:
            m_x = Concatenate(axis=-1, name=f"{prefix}feature_fusion")(feature_embeddings)
        else:
            m_x = feature_embeddings[0]
        m_x = BatchNormalization(name=f"{prefix}fusion_norm")(m_x)

        model_dim = self.sequential_model.internal_size
        m_x = Dense(model_dim, name=f"{prefix}input_projection")(m_x)

        # --- Transformer Block ---
        m_x = self.classification_head.apply_before_transformer(m_x, prefix)
        m_x = self.sequential_model.apply(m_x, prefix)
        m_x = self.classification_head.apply(m_x, prefix)


        # --- Projector MLP ---
        for layer_i, layer_size in enumerate(self.parameters.mlp_layer_sizes):
            m_x = Dense(layer_size, activation="relu", name=f"{prefix}classification_mlp_{layer_i}_{layer_size}")(m_x)
            if self.parameters.mlp_dropout > 0:
                m_x = Dropout(self.parameters.mlp_dropout)(m_x)

        # [Output 1] Feature Vector (Normalized for SupCon & Distillation)
        # 闂佸搫顦弲婊堟偡閳哄懎闂梺鍨儐娴溿倖绻涢幋鐐电煠闁衡偓?L2 Normalize闂備焦瀵х粙鎴︽儔婵傜鐏虫俊顖濆吹?SupCon ?闂備浇妫勯崐褰掑箖閸岀偑?濠电偠鎻紞鈧繛澶嬫礋瀵偊濡舵径瀣壄闂佸憡娲﹂崜姘辩懅闂備礁鎲￠〃鍡椕洪幋鐘电闁告稒娼欑憴锕傛煙鏉堝墽鐣辨繛?
        shared_representation = m_x
        feature_vector = Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="feature_norm")(shared_representation)

        # [Output 2] Binary Classification Head
        # 闂佸搫顦悧濠囧箰閹间礁鐭楅柛鈩冪☉?normalized features闂備焦瀵х粙鎴︽儗娴ｇ儤顐芥い鎾卞灩缁€鍕煟閹寸倖鎴犵懅 0~1 闂備焦鐪归崝宀€鈧凹鍙冮妴鍛邦樄?(Sigmoid)
        binary_pred = Dense(1, activation='sigmoid', name="binary_head")(shared_representation)

        all_inputs_list = []
        for name in self.model_input_spec.feature_names:
            if name in inputs_dict: all_inputs_list.append(inputs_dict[name])

        # 闂佸搫顦弲婊堝蓟閵娿儍娲冀椤剛鍠栭崹鎯х暦閸ワ絽浜鹃柣鎴ｅГ閺咁剚鎱ㄥ┑鍥ㄢ拹闁挎稒鍔欏畷鐔碱敆閸屾ê浠︽繝娈垮枟鐢帡宕愮拠宸晠?0-1), 闂備胶绮〃鍫ュ箠閹捐埖顐芥繛鎴欏灩鐟欙箓鏌熸潏鍓х暠婵炲拋?
        model = Model(inputs=all_inputs_list, outputs=[binary_pred, feature_vector], name="ids_model")
        return model

    # def train_incremental(self, model: keras.Model,
    #                       old_model: Optional[keras.Model],
    #                       epochs: int, batch_size: int, steps_per_epoch: int,
    #                       benign_class_idx: int = 0,
    #                       lr: float = 0.0001,
    #                       distill_lambda: float = 10.0,
    #                       supcon_lambda: float = 0.1,
    #                       temperature: float = 0.07,
    #                       old_class_ids: List[int] = None,
    #                       class_weight: Dict[int, float] = None,  # <--- [闂備礁鎼崐鐟邦熆濮椻偓璺柛?闂備浇顫夋禍浠嬪磿閺屻儱鏋佺憸鐗堝笒缁狙囨煕閹伴潧鏋涙繛鍛灲閹鈽夊▎妯荤暥闂佸憡鍩冮崑?
    #                       **kwargs) -> keras.Model:
    #     self.m = model
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    #     trainable_vars = self.m.trainable_variables
    #     bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #     mse_loss_fn = tf.keras.losses.MeanSquaredError()
    #     yielder = self.get_batch_yielder(batch_size=batch_size, for_training=True, balanced=True)
    #
    #     old_ids_tensor = None
    #     if old_class_ids is not None:
    #         old_ids_tensor = tf.constant(old_class_ids, dtype=tf.int32)
    #
    #     # [闂備礁鎼崐鐟邦熆濮椻偓璺柛?濠碘槅鍋呭妯尖偓姘煎墴钘濋柍鍝勬噺閸嬬娀鏌涢埄鍏╂垹绱為埀顒勬⒒娴ｇ懓绲荤紒澶屾暬閹繝鏁撻悩鑼吋闂佺顫夐崝鎴﹀焵?(濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞鐎ｎ亶鍤ら梺缁樕戣ぐ鍐嚕閸愯?class_weight)
    #     keys_tensor = None
    #     vals_tensor = None
    #     if class_weight is not None:
    #         # 闂佽绻愮换鎰涘Δ鍛仱鐟滄柨鐣峰Δ鈧～銏沪閸撗佸仏闂備胶顢婇鏍ь熆閳ь剟?Tensor 闂備礁鎼悮顐﹀磿閸欏鐝舵俊顖氬悑閸嬫﹢鏌曟繝蹇涙缂併劍宀搁弻?
    #         max_id = max(class_weight.keys())
    #         # 闂備礁鎲＄敮妤冪矙閹寸姷纾介柟鎯ь嚟閳绘棃鏌嶈閸撴氨绮欐径鎰垫晜闁割偁鍨圭粭锟犳煟閻橀亶妾烽柛濠冪墪椤洭宕奸妷锔惧帓閻庡箍鍎辩€氼垳?1.0 闂備焦鐪归崝宀€鈧凹鍙冨鎶芥偄閾忓湱?
    #         weights_arr = np.ones(max_id + 1, dtype=np.float32)
    #         for k, v in class_weight.items():
    #             weights_arr[k] = v
    #         weights_lookup = tf.constant(weights_arr)
    #
    #     for epoch in range(1, epochs + 1):
    #         with tqdm(range(steps_per_epoch), desc=f"Ep {epoch}/{epochs}", unit="step") as pbar:
    #             running_metrics = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "kd": 0.0, "acc": 0.0}
    #
    #             for step in pbar:
    #                 batch_x, batch_y = yielder.get_batch()
    #                 if not batch_x or batch_y.size == 0: continue
    #                 valid_mask = (batch_y != -1)
    #                 if not np.any(valid_mask): continue
    #
    #                 batch_x_tensors = [tf.convert_to_tensor(bx) for bx in batch_x]
    #                 batch_x_masked = [tf.boolean_mask(bx, valid_mask) for bx in batch_x_tensors]
    #                 y_true_fine = tf.boolean_mask(tf.constant(batch_y, dtype=tf.int32), valid_mask)
    #
    #                 # 濠电偛鐡ㄧ划宀勬嚐椤栫偛鍨傞幖杈剧悼鐏忕數绱撴担璇＄劷闁搞劏鍋愮槐?(0=Benign, 1=Attack)
    #                 y_true_binary = tf.cast(tf.not_equal(y_true_fine, benign_class_idx), tf.float32)
    #                 y_true_binary = tf.reshape(y_true_binary, (-1, 1))
    #
    #                 # [闂備礁鎼崐鐟邦熆濮椻偓璺柛?闂佽崵濮崇欢銈囨閺囥垺鍋╅柡鍕箞娴滃綊鏌熼幆褍鏆辨い?Batch 闂備焦鐪归崝宀€鈧凹鍙冮幃褔鏌嗗鍛彉濡炪倖甯婇悞锔剧礊閳ь剟?
    #                 sample_weights = None
    #                 if class_weight is not None:
    #                     # 闂佽娴烽幊鎾凰囬鐐茬煑闊洦绋戦拑鐔兼煏婢舵稓鐣辨い銉︾箞閺屻劌鈽夊▎鎺戭棟濡炪倐鏁崶褍鍤戝┑鐘诧工鐎氼厾?ID 闂佺儵鍓濈敮鎺楀箠鎼淬劌鍚规い鎾卞灪閸ゆ垿鏌涢幇鈺佸缂佺虎鍨堕弻銊モ槈濡偐鍔紓渚囧櫘閸ㄦ娊?1.0 (闂佸搫顦弲婊堟偡閳哄懎闂柣鎴ｆ绾剧粯鎱ㄥ鍡椾簽?clip 濠电儑绲藉ú锔炬崲閸愨晜?
    #                     safe_indices = tf.clip_by_value(y_true_fine, 0, len(weights_arr) - 1)
    #                     sample_weights = tf.gather(weights_lookup, safe_indices)
    #                     sample_weights = tf.reshape(sample_weights, (-1, 1))
    #
    #                 with tf.GradientTape() as tape:
    #                     prob_new, features_new = self.m(batch_x_masked, training=True)
    #
    #                     # 1. CE Loss (濠电偟顥愰崑鎰叏閹绢喖?sample_weights)
    #                     loss_ce = bce_loss_fn(y_true_binary, prob_new, sample_weight=sample_weights)
    #
    #                     # 2. SupCon Loss
    #                     loss_scl = supervised_contrastive_loss(features_new, y_true_fine, temperature)
    #
    #                     # 3. Distillation Loss
    #                     loss_kd = tf.constant(0.0)
    #                     if old_model is not None and distill_lambda > 0:
    #                         _, features_old = old_model(batch_x_masked, training=False)
    #
    #                         if old_ids_tensor is not None:
    #                             match_matrix = tf.equal(tf.reshape(y_true_fine, (-1, 1)),
    #                                                     tf.reshape(old_ids_tensor, (1, -1)))
    #                             is_old_sample = tf.reduce_any(match_matrix, axis=1)
    #                             feat_new_sel = tf.boolean_mask(features_new, is_old_sample)
    #                             feat_old_sel = tf.boolean_mask(features_old, is_old_sample)
    #                             if tf.shape(feat_new_sel)[0] > 0:
    #                                 loss_kd = mse_loss_fn(feat_old_sel, feat_new_sel)
    #                         else:
    #                             loss_kd = mse_loss_fn(features_old, features_new)
    #
    #                     total_loss = loss_ce + (supcon_lambda * loss_scl) + (distill_lambda * loss_kd)
    #
    #                 grads = tape.gradient(total_loss, trainable_vars)
    #                 optimizer.apply_gradients(zip(grads, trainable_vars))
    #
    #                 # Metrics
    #                 pred_binary = tf.cast(prob_new > 0.5, tf.float32)
    #                 acc = tf.reduce_mean(tf.cast(tf.equal(pred_binary, y_true_binary), tf.float32))
    #
    #                 running_metrics["loss"] += float(total_loss)
    #                 running_metrics["ce"] += float(loss_ce)
    #                 running_metrics["scl"] += float(loss_scl)
    #                 running_metrics["kd"] += float(loss_kd)
    #                 running_metrics["acc"] += float(acc)
    #
    #                 pbar.set_postfix({
    #                     "L": f"{running_metrics['loss'] / (step + 1):.3f}",
    #                     "CE": f"{running_metrics['ce'] / (step + 1):.3f}",
    #                     "SCL": f"{running_metrics['scl'] / (step + 1):.3f}",
    #                     "KD": f"{running_metrics['kd'] / (step + 1):.3f}",
    #                     "Acc": f"{running_metrics['acc'] / (step + 1):.3f}"
    #                 })
    #
    #     return self.m




    # 闂備礁鎼悧鍐磻閹剧粯鐓曟俊顖滃帶閺嬪孩銇勯锝呭?闂備礁鎼崐浠嬶綖婢跺本?
    # ---------------------------------------------------------------
    # 3. Train Incremental: 濠电姭鎷冮崨顓濈捕閻庤鐡曟ご鍝ョ矉瀹ュ骞㈡俊顖欑串缁辨岸妫呴銏″妞ゃ劌妫楄灒濠㈣泛鐬奸惌鍡涙煕濠靛棗顏悗姘冲皺閹叉悂鎳滈鈧悘顔尖攽椤旂虎鐒鹃摶鐐烘煃瑜滈崜鐔奉嚕?
    # ---------------------------------------------------------------
    #
    # [AA2018.py] -> train_incremental 闂備礁鎼崐浠嬶綖婢跺本?
    def train_incremental(self, model: keras.Model,
                          old_model: Optional[keras.Model],
                          epochs: int, batch_size: int, steps_per_epoch: int,
                          benign_class_idx: int = 0,
                          lr: float = 0.0001,
                          distill_lambda: float = 10.0,  # <--- 濠电儑绲藉ù鍌炲窗濡ゅ懎鏋佸┑鍌氭啞閸婇? 濠殿喗甯楃粙鎺椻€﹂崼銉晣濠电姵鑹剧壕褰掓煕韫囨挸鎮戦柡鍡╁幖?
                          supcon_lambda: float = 0.2,
                          temperature: float = 0.07,
                          old_class_ids: List[int] = None,  # <--- 濠电儑绲藉ù鍌炲窗濡ゅ懎鏋佸┑鍌氭啞閸婇? 闂備礁鎼崐鐟邦熆濮椻偓璺柛鎰靛枛閻鏌涚仦鍓р姇?
                          **kwargs) -> keras.Model:
        self.m = model
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        trainable_vars = self.m.trainable_variables
        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        yielder = self.get_batch_yielder(batch_size=batch_size, for_training=True, balanced=True)

        # 濠碘槅鍋呭妯尖偓姘煎墴钘濋柍鍝勬噺?old_class_ids ?Tensor闂備焦瀵х粙鎴炵附閺冣偓缁傚秹骞掗幘鍓佹嚌闂佸壊鐓堥崰妤呯叕椤掑嫭鐓曢柕澶涚畱缁€鍐┿亜閹捐櫕鎲哥紒顔款嚙椤啴骞戦幇顒€绠戝┑鐐舵彧缂嶁偓婵炲娲熷?
        old_ids_tensor = None
        if old_class_ids is not None:
            old_ids_tensor = tf.constant(old_class_ids, dtype=tf.int32)

        for epoch in range(1, epochs + 1):
            with tqdm(range(steps_per_epoch), desc=f"Ep {epoch}/{epochs}", unit="step") as pbar:
                running_metrics = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "kd": 0.0, "acc": 0.0}

                for step in pbar:
                    batch_x, batch_y = yielder.get_batch()
                    if not batch_x or batch_y.size == 0: continue
                    valid_mask = (batch_y != -1)
                    if not np.any(valid_mask): continue

                    batch_x_tensors = [tf.convert_to_tensor(bx) for bx in batch_x]
                    batch_x_masked = [tf.boolean_mask(bx, valid_mask) for bx in batch_x_tensors]
                    y_true_fine = tf.boolean_mask(tf.constant(batch_y, dtype=tf.int32), valid_mask)

                    # 濠电偛鐡ㄧ划宀勬嚐椤栫偛鍨傞幖杈剧悼鐏忕數绱撴担璇＄劷闁搞劏鍋愮槐?
                    y_true_binary = tf.cast(tf.not_equal(y_true_fine, benign_class_idx), tf.float32)
                    y_true_binary = tf.reshape(y_true_binary, (-1, 1))

                    with tf.GradientTape() as tape:
                        # 闂備礁鎼崐缁樻櫠濡ゅ嫬鏆為梻浣规偠閸斿矂顢栭崨顔芥珷闁绘顕х憴?
                        prob_new, features_new = self.m(batch_x_masked, training=True)

                        # 1. CE Loss
                        loss_ce = bce_loss_fn(y_true_binary, prob_new)

                        # 2. Binary SupCon Loss: group all attacks together against Benign.
                        y_supcon_binary = tf.cast(tf.reshape(y_true_binary, (-1,)), tf.int32)
                        loss_scl = supervised_contrastive_loss(features_new, y_supcon_binary, temperature)

                        # 3. Distillation Loss (濠电偛顕慨鎾箠濮椻偓楠炲繘鏁嶉崟銊ヤ壕濞达絽鍠氬鎰繆椤愵偄浜為柣?
                        loss_kd = tf.constant(0.0)
                        if old_model is not None and distill_lambda > 0:
                            _, features_old = old_model(batch_x_masked, training=False)

                            # --- 闂備礁鎼粔鍫曗€﹂崼銏㈢处闁绘劦鍓涢埞宥嗐亜閺冨洤袚闁哄棭鍙冮弻銊モ槈濞嗗繐鎽甸梺閫炲苯澧い锔藉閳ь剚鍝庨崝鎴︾嵁閳ь剛鎲搁幋锕€绠查柍褜鍓欒灋?---
                            if old_ids_tensor is not None:
                                # 闂備礁鎲＄敮鍥磹閺嶎厼钃熼柛銉仢閹烘洜鐤€閹兼番鍔屽▓婵嬫⒑閸濆嫮效闁衡偓閸楃儑鑰垮ù鐓庣摠閸庡秹鏌涢弴銊ュ闁搞劏鍋愮槐鎺戔槈濮橆兛澹曞┑鈥冲级閹倸鐣烽妷鈺傛櫇闁逞屽墴閹牆螣閼姐値?old_class_ids
                                # shape: (batch_size, 1) == (1, n_old_ids) -> (batch_size, n_old_ids)
                                match_matrix = tf.equal(tf.reshape(y_true_fine, (-1, 1)),
                                                        tf.reshape(old_ids_tensor, (1, -1)))
                                # 闂備礁鎲￠悷顖涚閿濆鈧懘鏁冮崒姘彉婵犮垼娉涢ˇ閬嶎敂閸楃偐妲堥柟鍨暕缁ㄥ鎮樿箛锝呮珝闁哄苯锕ら鑹扮疀閺囩媭妲烽梺鑽ゅТ濞层垽宕曢弻銉﹀亱闂侇剙绉寸€氬銇勯幒鍡椾壕濠殿喗菧閸旀垵顕ｆ导鎼晬婵炴垶鐟辩槐婊呯磽?
                                is_old_sample = tf.reduce_any(match_matrix, axis=1)

                                # 濠电偛顕慨鎾箠鎼淬們浜归柛銉墮缁€鍕煕濠靛棗顏繛纭风到闇夋繝濠傚暙椤掋垺淇婇顐簽闁绘岸绠栭幊锟犲Χ閸℃瑦娈搁梻浣虹帛椤ㄥ牓骞婇幘鑸殿偨婵炴垶鐟︽禍銈嗙箾閹寸偟鐭ら柡?MSE 闂佽崵濮崇欢銈囨閺囥垺?
                                feat_new_sel = tf.boolean_mask(features_new, is_old_sample)
                                feat_old_sel = tf.boolean_mask(features_old, is_old_sample)

                                # 闂佽娴烽幊鎾凰囬鐐茬煑闊洦鎷瑰☉銏犵劦妞ゆ帒瀚拑鐔兼煏婢舵ê鐏ｇ紒鎰殔閳规垿鍨鹃崘娴嬪亾瑜版帗鍊跺璺侯儐娴溿倖绻涢崱妤冪闁汇劍鍨剁换娑氫沪缁嬪灝鈷夊┑鈽嗗亽閸嬪棝鎮崜褏纾兼俊顖滃帶椤ュ酣姊洪崫鍕⒈闁告挶鍔庡Σ鎰煶椤s ?0
                                if tf.shape(feat_new_sel)[0] > 0:
                                    loss_kd = mse_loss_fn(feat_old_sel, feat_new_sel)
                            else:
                                # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞婵炵偓鐓㈤梺鐐藉劥鐏忔瑩?ID闂備焦瀵х粙鎴︽嚐椤栨凹娼╅柕濞р偓閸嬫挻鎷呴崘鐐秷婵犫拃鍐х€殿喖鐏氬鍕偓锝庡墮椤ュ酣姊洪崫鍕⒈闁告挻绻堥獮鎴﹀焵椤掆偓?(Fallback)
                                loss_kd = mse_loss_fn(features_old, features_new)

                        # 闂備浇顕栭崜娑氬垝椤栨氨鏆︽い鏍ㄧ☉?
                        total_loss = loss_ce + (supcon_lambda * loss_scl) + (distill_lambda * loss_kd)

                    grads = tape.gradient(total_loss, trainable_vars)
                    optimizer.apply_gradients(zip(grads, trainable_vars))

                    # 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘缁€鍕煕閳╁啰鎲块柍褜鍓氶敃銏ゅ箖?
                    pred_binary = tf.cast(prob_new > 0.5, tf.float32)
                    acc = tf.reduce_mean(tf.cast(tf.equal(pred_binary, y_true_binary), tf.float32))

                    running_metrics["loss"] += float(total_loss)
                    running_metrics["ce"] += float(loss_ce)
                    running_metrics["scl"] += float(loss_scl)
                    running_metrics["kd"] += float(loss_kd)
                    running_metrics["acc"] += float(acc)

                    div = step + 1
                    # 闁诲孩顔栭崰鏍箹椤愶箑鏄ユ俊銈勭劍娴溿倖绻涢幋鐏活亪顢欐繝鍥ㄧ厸濠㈣泛妫楁晶鍙変繆椤栨凹妲虹紒?
                    pbar.set_postfix({
                        "L": f"{running_metrics['loss'] / div:.3f}",
                        "B": f"{running_metrics['ce'] / div:.2f}",
                        "S": f"{running_metrics['scl'] / div:.2f}",
                        "K": f"{running_metrics['kd'] / div:.3f}",
                        "A": f"{running_metrics['acc'] / div:.2f}"
                    })
                    # 闂佽姘﹂～澶屽枈瀹ュ拑鑰块柨鐔哄Т缁犮儵鏌熼幆褍鏆辩紒?
                    if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                        avg_l = running_metrics['loss'] / div
                        avg_b = running_metrics['ce'] / div
                        avg_s = running_metrics['scl'] / div
                        avg_k = running_metrics['kd'] / div
                        avg_a = running_metrics['acc'] / div
                        pbar.write(
                            f"Ep {epoch} [{step + 1}] L:{avg_l:.3f} | CE:{avg_b:.3f} | SCL:{avg_s:.3f} | KD:{avg_k:.3f} | Acc:{avg_a:.3f}")

        return self.m

    # def train_incremental(self, model: keras.Model,
    #                       old_model: Optional[keras.Model],
    #                       epochs: int, batch_size: int, steps_per_epoch: int,
    #                       benign_class_idx: int = 0,
    #                       lr: float = 0.0001,
    #                       distill_lambda: float = 0.5,
    #                       supcon_lambda: float = 0.1,
    #                       temperature: float = 0.07,
    #                       **kwargs) -> keras.Model:
    #     self.m = model
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr,clipnorm=1.0)
    #     trainable_vars = self.m.trainable_variables
    #
    #     # Binary Cross Entropy (闂備焦妞垮鍧楀礉鐎ｎ剝濮虫い鎺戝缁€鍡涙煕閳╁啰鎳嗘い?
    #     bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    #     # MSE (闂備焦妞垮鍧楀礉鐎ｎ剝濮虫い鎺戝閸嬪鏌ｉ悢绋款棆缂佹劖鐟╅弻锕傚磼濞戞瑥鈧劖?
    #     mse_loss_fn = tf.keras.losses.MeanSquaredError()
    #
    #     yielder = self.get_batch_yielder(batch_size=batch_size, for_training=True, balanced=True)
    #
    #     # [AA2018.py] - train_incremental 闂備礁鎼崐浠嬶綖婢跺本鍏滈柛顐ｆ礀缁€鍐煕濞戞瑦缍戦柛?
    #     # 闂備礁鎼ú锔锯偓绗涘啰鏆﹂柛娆忣槺椤╃兘骞栧ǎ顒€鈧绮昏ぐ鎺撯拺闁圭粯甯熺€氭壆绱掓潏銊ョ仸闁逛究鍔庨埀顒婄秵閸撴岸鎮鹃崡鐏诲綊鎮崨顖涘仹?
    #
    #     for epoch in range(1, epochs + 1):
    #         # desc 闂備礁鎲￠悷顖涚濠靛柈鐔稿緞瀹€鈧惌?Ep x/x闂備焦瀵х粙鎴﹀嫉椤掍椒绻嗛柛婵勫劤閻碍绻涘顔荤凹闁挎稓鍠栭弻锝夋晲閸℃瑦鍣紓渚囧枙閸楀啿鐣烽妷锔藉劅闁靛闄勯妴鍐⒑濮瑰洤濡奸悗姘煎幘閳ь剙鐏氬畝绋款嚕?
    #         with tqdm(range(steps_per_epoch), desc=f"Ep {epoch}/{epochs}", unit="step") as pbar:
    #             # 缂傚倷绶氱涵鎼佸垂閻㈠壊鏁囬柟闂寸缁犱即鏌涢妷锝呭闁搞劌銈搁弻锝夛綖椤掆偓婵¤櫣绱掗…鎴濆⒋鐎殿喕鍗冲畷婊嗩槹濞寸姵绋掗〃銉╂倷閹绘帗姣愬┑鐐靛帶闁帮絽?
    #             running_metrics = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "kd": 0.0, "acc": 0.0}
    #
    #             for step in pbar:
    #                 batch_x, batch_y = yielder.get_batch()
    #                 # 1. 闂佽娴烽幊鎾凰囬鐐茬煑闊洦鎷瑰☉銏犵劦妞ゆ帒瀚拑? 闂備線鈧稓绁锋い顐㈩樀椤㈡洟宕楅懖鈹炬灃闂佹儳绻愰崢婊堝极?
    #                 if not batch_x or batch_y.size == 0: continue
    #
    #                 # 2. 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦粻鍐差熆閼搁潧濮堝┑顔界矋缁绘稒寰勭€ｎ偆顦ㄩ梺缁樼壄缁茶法?(-1)
    #                 valid_mask = (batch_y != -1)
    #                 if not np.any(valid_mask): continue
    #
    #                 # 3. 闂備礁鎲￠崹闈浳涘Δ?Tensor
    #                 batch_x_tensors = [tf.convert_to_tensor(bx) for bx in batch_x]
    #                 # 濠电偠鎻紞鈧繛澶嬫礋?boolean_mask 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦弸渚€鏌ｅΔ鈧悧鍡欑矈?
    #                 batch_x_masked = [tf.boolean_mask(bx, valid_mask) for bx in batch_x_tensors]
    #                 y_true_fine = tf.boolean_mask(tf.constant(batch_y, dtype=tf.int32), valid_mask)
    #
    #                 # 4. 闂備礁鎲＄敮妤€锕㈤崷顓犵闁哄啠鍋撶紒鍌氱Ф娴狅箓鎸婃径灞稿亾閼姐倗纾兼い鎰╁€楃敮娑㈡煟閺嶇數鎮肩紒? Benign(0) -> 0.0, Attack(>0) -> 1.0
    #                 y_true_binary = tf.cast(tf.not_equal(y_true_fine, benign_class_idx), tf.float32)
    #                 y_true_binary = tf.reshape(y_true_binary, (-1, 1))  # (N, 1)
    #
    #                 with tf.GradientTape() as tape:
    #                     # Forward Pass (Current Model) -> [prob, features]
    #                     prob_new, features_new = self.m(batch_x_masked, training=True)
    #
    #                     # A. Binary Cross Entropy Loss
    #                     loss_ce = bce_loss_fn(y_true_binary, prob_new)
    #
    #                     # B. Supervised Contrastive Loss (濠电偠鎻紞鈧繛澶嬫礋瀵偊濡堕崶鈺冿紲闂佺硶鍓濋悷褔鎮￠崼銉ョ閺夊牓绠栧顕€鏌ｉ弽鐢垫偧?y_true_fine)
    #                     loss_scl = supervised_contrastive_loss(features_new, y_true_fine, temperature)
    #
    #                     # C. Feature Distillation
    #                     loss_kd = tf.constant(0.0)
    #                     if old_model is not None and distill_lambda > 0:
    #                         # 闂備礁鎼崬鏌ュ礋閳ь剚绗熼埀顒€鐣烽崷顓涘亾閿濆倹娅囬柣鐔哥箞瀵爼鍩￠崒婊庣伇?training=True (Dropout 闂備胶顭堢换鎴炵箾婵犲伣?
    #                         _, features_old = old_model(batch_x_masked, training=False)
    #                         loss_kd = mse_loss_fn(features_old, features_new)
    #
    #                     # Total Loss
    #                     total_loss = loss_ce + (supcon_lambda * loss_scl) + (distill_lambda * loss_kd)
    #
    #                 # Backprop
    #                 grads = tape.gradient(total_loss, trainable_vars)
    #                 optimizer.apply_gradients(zip(grads, trainable_vars))
    #
    #                 # --- 闂佽崵濮崇欢銈囨閺囥垺鍋╅柤濮愬€栨刊鍏笺亜閹捐泛啸闁活厼锕濠氬幢濡や緡妫涚紓浣诡殔閸婂潡骞嗛崘顔肩妞ゆ劑鍊栧鏍⒑閸濆嫬鏆㈡繛澶嬬洴瀹曟娊顢涘鍕幑濡炪倖姊归崕鎶芥嚍?---
    #                 pred_binary = tf.cast(prob_new > 0.5, tf.float32)
    #                 acc = tf.reduce_mean(tf.cast(tf.equal(pred_binary, y_true_binary), tf.float32))
    #
    #                 # 闂備礁鎼ú銈夋偤閵娾晛钃熷┑鐘插椤洘銇勯弽銊х闁告埃鍋撻梻浣告贡椤牆煤閿曞倹?
    #                 running_metrics["loss"] += float(total_loss)
    #                 running_metrics["ce"] += float(loss_ce)
    #                 running_metrics["scl"] += float(loss_scl)
    #                 running_metrics["kd"] += float(loss_kd)
    #                 running_metrics["acc"] += float(acc)
    #
    #                 div = step + 1
    #
    #                 # === 闂傚倸鍊稿ú鐘诲磻閹惧瓨?1: 闂佸搫顦弲婊呯矙閹寸姭鍋撶憴鍕枙?set_postfix (闁诲孩顔栭崰鏍箹椤愶箑鏄ユ俊銈勮兌閳瑰秵绻濋棃娑氬婵炲牆鐭傞弻鐔哄枈濡桨澹曢梻浣告惈閻楀棝藝閻㈢违? ===
    #                 # 闂備礁鎲￠〃鍛村储閺嶃劎绠鹃柛銉㈡櫃濞岊亪鐓崶銊﹀暗妞ゆ柨娴风槐鎺楀箛椤掆偓椤忣剚銇勯顒傜暤妤犵偛顑夐、鏃堝川椤旀儳濮洪梻浣瑰缁嬫垿鎯屾担鐑樺床闁哄稁鍘介悡鍌炴倶閻愯泛浜归柣锕€绉归弻娑滅疀鐎ｎ亜顫囬梺绋块閻°劎鎹㈤崨瀛樺€烽柛鎾茶兌椤旀帡?
    #                 pbar.set_postfix({
    #                     "L": f"{running_metrics['loss'] / div:.3f}",
    #                     "B": f"{running_metrics['ce'] / div:.2f}",
    #                     "S": f"{running_metrics['scl'] / div:.2f}",
    #                     "K": f"{running_metrics['kd'] / div:.3f}",
    #                     "A": f"{running_metrics['acc'] / div:.2f}"
    #                 })
    #
    #                 # === 闂傚倸鍊稿ú鐘诲磻閹惧瓨?2: 闂備礁鎲￠懝楣冨嫉椤掆偓椤啴宕掑☉娆忣€撻柣鐘充航閸斿秹?Print (濠电偠鎻紞鈧繛澶嬫礋?pbar.write 闂傚倸鍊搁崯顐㈩嚕閸洦鏁嗘繛鎴欏灩缁€鍡欌偓鐟板婢瑰棗? ===
    #                 # ?20 婵犳鍠楃缓鍧楀磿閺屻儱鍨傞柕濞炬櫆閸も偓闂佸憡渚楅崢浠嬬叕椤掑嫭鐓涢柛顐ｈ壘娴滈箖姊洪崨濠呭缂併劏鍋愮划鈺呭级閹搭厽妗ㄩ梺闈涱檧闂勫嫰藝瑜旈弻銊モ槈濡粯鎷辨繝鈷€宥囩暤鐎规洩绻濆鍓佹崉閵婏附顔呴梻浣芥〃缁€渚€顢氶鐐╂灁闁瑰鍋熼埢鏃堟煃瑜滈崜娑㈠箖閹殿喕娌柣鎰靛墾缁辨岸?
    #                 # 闂佸搫顦弲婊堟偡閿曞倹鍋嬮梺顒€绉寸涵鈧梺绋跨箰閹冲酣?set_postfix 闂佽崵鍋為崙褰掑磻閸涙潙鏄ラ柛娑欐綑濡ɑ銇勯幘鍗炲缂佲偓婢跺⊕褰掓晲閸ヮ灝锛勭磼閳ュ啿鏆ｉ柟铏殜閹墽浠﹂崜褏鏆梻浣告啞鐢繝銆傞敂鍓х當闁稿瞼鍋為ˉ鍫ユ煏韫囧鐏柣?Log
    #                 if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
    #                     avg_l = running_metrics['loss'] / div
    #                     avg_b = running_metrics['ce'] / div
    #                     avg_s = running_metrics['scl'] / div
    #                     avg_k = running_metrics['kd'] / div
    #                     avg_a = running_metrics['acc'] / div
    #
    #                     log_msg = (f"Step [{step + 1}/{steps_per_epoch}] "
    #                                f"Loss: {avg_l:.3f} | "
    #                                f"CE: {avg_b:.3f} | "
    #                                f"SCL: {avg_s:.3f} | "
    #                                f"KD: {avg_k:.3f} | "
    #                                f"Acc: {avg_a:.3f}")
    #
    #                     # pbar.write 濠电偞娼欓崥瀣枈瀹ュ棔绻嗛柛婵勫劗閸嬫挸鈽夊▎妯荤暦婵犫拃宥囩暤鐎规洩绻濆鍫曞箰鎼淬倕鎯堥梺鍝勵槴閺呮粎绮欓幋鐘亾鐟欏嫬鈻曠€殿喖鐖奸崹楣冩嚑椤掑倹鍟伴梻浣告惈閸婅鈻嶉姀銏☆潟婵犻潧娲ㄩ埢鏃傗偓骞垮劚鐎氼喚绮ｅΔ鍛厵闁割煈鍋呯亸顐︽偣閹般劌澧叉繛鑹邦嚙閳藉螣娓氼垰鏁介梻?
    #                     pbar.write(log_msg)
    #
    #     return self.m


        # # [AA2018.py] - train_incremental 闂備礁鎼崐浠嬶綖婢跺本鍏滈柛顐ｆ礀缁€鍐煕濞戞瑦缍戦柛?
        # for epoch in range(1, epochs + 1):
        #     # 濠电儑绲藉ù鍌炲窗濡ゅ懎?: ?epoch 濠电儑绲藉ú鐘诲礈濠靛洤顕遍柛娑卞灣閻捇鎮归崶銊ョ祷?desc 濠电偞鍨堕幖鈺呭矗韫囨洘顫曟繝闈涱儐閸ゅ秹鏌涚仦鍓х煁?postfix 闂備焦鐪归崝宀€鈧凹鍣ｉ崺鐐哄冀椤撶啿?
        #     with tqdm(range(steps_per_epoch), desc=f"Ep {epoch}/{epochs}", unit="step") as pbar:
        #         # 缂傚倷绶氱涵鎼佸垂閻㈠壊鏁囬柟闂寸缁犱即鏌涢妷锝呭闁搞劌銈搁弻锝夛綖椤掆偓婵¤櫣绱掗…鎴濆⒋鐎殿喕鍗冲畷婊嗩槹濞寸姵绋掗〃銉╂倷閹绘帗姣愬┑鐐靛帶闁帮絽?
        #         running_metrics = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "kd": 0.0, "acc": 0.0}
        #
        #         for step in pbar:
        #             batch_x, batch_y = yielder.get_batch()
        #             # 1. 闂佽娴烽幊鎾凰囬鐐茬煑闊洦鎷瑰☉銏犵劦妞ゆ帒瀚拑? 闂備線鈧稓绁锋い顐㈩樀椤㈡洟宕楅懖鈹炬灃闂佹儳绻愰崢婊堝极?
        #             if not batch_x or batch_y.size == 0: continue
        #
        #             # 2. 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦粻鍐差熆閼搁潧濮堝┑顔界矋缁绘稒寰勭€ｎ偆顦ㄩ梺缁樼壄缁茶法?(-1)
        #             valid_mask = (batch_y != -1)
        #             if not np.any(valid_mask): continue
        #
        #             # 3. 闂備礁鎲￠崹闈浳涘Δ?Tensor
        #             batch_x_tensors = [tf.convert_to_tensor(bx) for bx in batch_x]
        #             # 濠电偠鎻紞鈧繛澶嬫礋?boolean_mask 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦弸渚€鏌ｅΔ鈧悧鍡欑矈?
        #             batch_x_masked = [tf.boolean_mask(bx, valid_mask) for bx in batch_x_tensors]
        #             y_true_fine = tf.boolean_mask(tf.constant(batch_y, dtype=tf.int32), valid_mask)
        #
        #             # 4. 闂備礁鎲＄敮妤€锕㈤崷顓犵闁哄啠鍋撶紒鍌氱Ф娴狅箓鎸婃径灞稿亾閼姐倗纾兼い鎰╁€楃敮娑㈡煟閺嶇數鎮肩紒? Benign(0) -> 0.0, Attack(>0) -> 1.0
        #             y_true_binary = tf.cast(tf.not_equal(y_true_fine, benign_class_idx), tf.float32)
        #             y_true_binary = tf.reshape(y_true_binary, (-1, 1))  # (N, 1)
        #
        #             with tf.GradientTape() as tape:
        #                 # Forward Pass (Current Model) -> [prob, features]
        #                 prob_new, features_new = self.m(batch_x_masked, training=True)
        #
        #                 # A. Binary Cross Entropy Loss
        #                 loss_ce = bce_loss_fn(y_true_binary, prob_new)
        #
        #                 # B. Supervised Contrastive Loss (濠电偠鎻紞鈧繛澶嬫礋瀵偊濡堕崶鈺冿紲闂佺硶鍓濋悷褔鎮￠崼銉ョ閺夊牓绠栧顕€鏌ｉ弽鐢垫偧?y_true_fine)
        #                 loss_scl = supervised_contrastive_loss(features_new, y_true_fine, temperature)
        #
        #                 # C. Feature Distillation
        #                 loss_kd = tf.constant(0.0)
        #                 if old_model is not None and distill_lambda > 0:
        #                     # 闂備礁鎼崬鏌ュ礋閳ь剚绗熼埀顒€鐣烽崷顓涘亾閿濆倹娅囬柣鐔哥箞瀵爼鍩￠崒婊庣伇?training=True (Dropout 闂備胶顭堢换鎴炵箾婵犲伣?
        #                     _, features_old = old_model(batch_x_masked, training=False)
        #                     loss_kd = mse_loss_fn(features_old, features_new)
        #
        #                 # Total Loss
        #                 total_loss = loss_ce + (supcon_lambda * loss_scl) + (distill_lambda * loss_kd)
        #
        #             # Backprop
        #             grads = tape.gradient(total_loss, trainable_vars)
        #             optimizer.apply_gradients(zip(grads, trainable_vars))
        #
        #             # --- 闂佽崵濮崇欢銈囨閺囥垺鍋╅柤濮愬€栨刊鍏笺亜閹捐泛啸闁活厼锕濠氬幢濡や緡妫涚紓浣诡殔閸婂潡骞嗛崘顔肩妞ゆ劑鍊栧鏍⒑閸濆嫬鏆㈡繛澶嬬洴瀹曟娊顢涘鍕幑濡炪倖姊归崕鎶芥嚍?---
        #             pred_binary = tf.cast(prob_new > 0.5, tf.float32)
        #             acc = tf.reduce_mean(tf.cast(tf.equal(pred_binary, y_true_binary), tf.float32))
        #
        #             # 闂備礁鎼ú銈夋偤閵娾晛钃熷┑鐘插暞娴溿倖绻涢幋鐏活亪顢欐繝鍥ㄧ厸濠㈣泛妫楁晶鍙変繆椤愮喐娅堢紒?
        #             running_metrics["loss"] += float(total_loss)
        #             running_metrics["ce"] += float(loss_ce)
        #             running_metrics["scl"] += float(loss_scl)
        #             running_metrics["kd"] += float(loss_kd)
        #             running_metrics["acc"] += float(acc)
        #
        #             # --- 濠电儑绲藉ù鍌炲窗濡ゅ懎?: 缂傚倷鑳舵慨顓㈠磻閹剧粯?Key 濠电偛顕慨浼村磿闂堟党锝囨嫚瀹割喗妗ㄩ梺闈涢獜缁茶法绮绘繝姘閺夊牓绠栧顔界箾绾板崬鐏︽鐐村灴婵偓闁挎繂瀛╅弬?---
        #             # L=Loss, B=BCE, S=SupCon, K=KD, A=Acc
        #             div = step + 1
        #             pbar.set_postfix({
        #                 "L": f"{running_metrics['loss'] / div:.3f}",
        #                 "B": f"{running_metrics['ce'] / div:.2f}",
        #                 "S": f"{running_metrics['scl'] / div:.2f}",
        #                 "K": f"{running_metrics['kd'] / div:.3f}",
        #                 "A": f"{running_metrics['acc'] / div:.2f}"
        #             })

        # for epoch in range(1, epochs + 1):
        #     with tqdm(range(steps_per_epoch), desc=f"Ep {epoch}/{epochs}", unit="step") as pbar:
        #         # 缂傚倷绶氱涵鎼佸垂閻㈠壊鏁囬柟闂寸缁犱即鏌涢妷锝呭闁搞劌銈搁弻锝夛綖椤掆偓婵¤櫣绱掗…鎴濆⒋鐎殿喕鍗冲畷婊嗩槹濞寸姵绋掗〃銉╂倷閹绘帗姣愬┑鐐靛帶闁帮絽?
        #         running_metrics = {"loss": 0.0, "ce": 0.0, "scl": 0.0, "kd": 0.0, "acc": 0.0}
        #
        #         for step in pbar:
        #             batch_x, batch_y = yielder.get_batch()
        #             # 1. 闂佽娴烽幊鎾凰囬鐐茬煑闊洦鎷瑰☉銏犵劦妞ゆ帒瀚拑? 闂備線鈧稓绁锋い顐㈩樀椤㈡洟宕楅懖鈹炬灃闂佹儳绻愰崢婊堝极?
        #             if not batch_x or batch_y.size == 0: continue
        #
        #             # 2. 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦粻鍐差熆閼搁潧濮堝┑顔界矋缁绘稒寰勭€ｎ偆顦ㄩ梺缁樼壄缁茶法?(-1)
        #             valid_mask = (batch_y != -1)
        #             if not np.any(valid_mask): continue
        #
        #             # 3. 闂備礁鎲￠崹闈浳涘Δ?Tensor
        #             batch_x_tensors = [tf.convert_to_tensor(bx) for bx in batch_x]
        #             # 濠电偠鎻紞鈧繛澶嬫礋?boolean_mask 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦弸渚€鏌ｅΔ鈧悧鍡欑矈?
        #             batch_x_masked = [tf.boolean_mask(bx, valid_mask) for bx in batch_x_tensors]
        #             y_true_fine = tf.boolean_mask(tf.constant(batch_y, dtype=tf.int32), valid_mask)
        #
        #             # 4. 闂備礁鎲＄敮妤€锕㈤崷顓犵闁哄啠鍋撶紒鍌氱Ф娴狅箓鎸婃径灞稿亾閼姐倗纾兼い鎰╁€楃敮娑㈡煟閺嶇數鎮肩紒? Benign(0) -> 0.0, Attack(>0) -> 1.0
        #             y_true_binary = tf.cast(tf.not_equal(y_true_fine, benign_class_idx), tf.float32)
        #             y_true_binary = tf.reshape(y_true_binary, (-1, 1))  # (N, 1)
        #
        #             with tf.GradientTape() as tape:
        #                 # Forward Pass (Current Model) -> [prob, features]
        #                 prob_new, features_new = self.m(batch_x_masked, training=True)
        #                 # prob_new shape should be (N, 1)
        #
        #                 # A. Binary Cross Entropy Loss
        #                 loss_ce = bce_loss_fn(y_true_binary, prob_new)
        #
        #                 # B. Supervised Contrastive Loss (濠电偠鎻紞鈧繛澶嬫礋瀵偊濡堕崶鈺冿紲闂佺硶鍓濋悷褔鎮￠崼銉ョ閺夊牓绠栧顕€鏌ｉ弽鐢垫偧?y_true_fine)
        #                 loss_scl = supervised_contrastive_loss(features_new, y_true_fine, temperature)
        #
        #                 # C. Feature Distillation
        #                 loss_kd = tf.constant(0.0)
        #                 if old_model is not None and distill_lambda > 0:
        #                     # 闂備礁鎼崬鏌ュ礋閳ь剚绗熼埀顒€鐣烽崷顓涘亾閿濆倹娅囬柣鐔哥箞瀵爼鍩￠崒婊庣伇?training=True (Dropout 闂備胶顭堢换鎴炵箾婵犲伣?
        #                     _, features_old = old_model(batch_x_masked, training=False)
        #                     loss_kd = mse_loss_fn(features_old, features_new)
        #
        #                 # Total Loss
        #                 total_loss = loss_ce + (supcon_lambda * loss_scl) + (distill_lambda * loss_kd)
        #
        #             # Backprop
        #             grads = tape.gradient(total_loss, trainable_vars)
        #             optimizer.apply_gradients(zip(grads, trainable_vars))
        #
        #             # --- 闂佽崵濮崇欢銈囨閺囥垺鍋╅柤濮愬€栨刊鍏笺亜閹捐泛啸闁活厼锕濠氬幢濡や緡妫涚紓浣诡殔閸婂潡骞嗛崘顔肩妞ゆ劑鍊栧鏍⒑閸濆嫬鏆㈡繛澶嬬洴瀹曟娊顢涘鍕幑濡炪倖姊归崕鎶芥嚍?(Debug? ---
        #             pred_binary = tf.cast(prob_new > 0.5, tf.float32)
        #             acc = tf.reduce_mean(tf.cast(tf.equal(pred_binary, y_true_binary), tf.float32))
        #
        #             # 闂備礁鎼ú銈夋偤閵娾晛钃熷┑鐘插暞娴溿倖绻涢幋鐏活亪顢欐繝鍥ㄧ厸濠㈣泛妫楁晶鍙変繆椤愮喐娅堢紒?
        #             running_metrics["loss"] += float(total_loss)
        #             running_metrics["ce"] += float(loss_ce)
        #             running_metrics["scl"] += float(loss_scl)
        #             running_metrics["kd"] += float(loss_kd)
        #             running_metrics["acc"] += float(acc)
        #
        #             # 闂備礁鎼€氼剚鏅舵禒瀣︽慨妯哄瀹撲線鏌ц箛锝呬簽闁规灚鍔戦弻娑橆潩椤掍焦宕冲┑鐐茬墛閸ㄥ潡鎮伴鈧畷姗€宕ｆ径宀€校?
        #             div = step + 1
        #             pbar.set_postfix({
        #                 "L": f"{running_metrics['loss'] / div:.3f}",
        #                 "B": f"{running_metrics['ce'] / div:.3f}",
        #                 "S": f"{running_metrics['scl'] / div:.3f}",
        #                 "K": f"{running_metrics['kd'] / div:.3f}",
        #                 "Acc": f"{running_metrics['acc'] / div:.3f}"  # 闂備礁鎼崐鐟邦熆濮椻偓? 闂佽崵濮崇欢锟犲储閸撗冨灊鐎光偓閸曨偆顦╅梺绯曞墲閻熴垽鍩€椤掍緤鑰块柟?
        #             })

        # return self.m

    # ---------------------------------------------------------------
    # 4. Evaluation (Binary Metrics) with Detail Debug
    # ---------------------------------------------------------------

        # [濠电儑绲藉ù鍌炲窗濡ゅ懎?5] main_fitday1.py -> IncrementalLearner ?-> evaluate_multiclass (闂備礁鎼ú锔锯偓绗涘啰鏆﹂柡灞诲劚閸屻劑鏌ｉ弬鎸庡暈闂傚嫬绉归弻?evaluate_binary)

    def evaluate_multiclass(self, dataset_name, dataset_df, dataset_spec, class_to_idx):
        print(f"\n--- Evaluation [{dataset_name}] (Multi-class) ---")

        if dataset_df.empty: return {}

        # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?
        handler = FlowTransformer(self.pre_processor, **self.ft_components)
        handler.m = self.model
        handler.model_input_spec = self.global_input_spec
        handler.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
                             evaluation_percent=0.0, fit_preprocessor=False)  # Eval=0.0 闂備胶顭堢换鍫ュ礉閹达箑纾块柟缁㈠枟閸嬨劑鏌曟繛鍨偓妤呮嚌閻愵剛绠鹃柡澶嬪焾閸庢捇?

        # 2. 濠碘槅鍋呭妯尖偓姘煎幘?
        y_true_list, y_pred_list = [], []
        yielder = handler.get_batch_yielder(512, for_training=False, balanced=False)

        while True:
            bx, by = yielder.get_batch()
            if not bx: break
            prob, _ = self.model.predict(bx, verbose=0)

            # 濠电姰鍨奸崺鏍嚌妤ｅ啫鍨傞幖杈剧悼鐏忛潧霉閸忚偐鎳呯紒鎰殜?argmax
            pred_cls = np.argmax(prob, axis=1)

            y_true_list.extend(by)
            y_pred_list.extend(pred_cls)

        handler.clear_memory()
        del handler
        gc.collect()

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦猾宥夋煟濡寧鐝繛鍛箻閺屸剝鎷呴崫鍕垫毉?
        mask = (y_true != -1)
        y_true, y_pred = y_true[mask], y_pred[mask]

        # 3. 闂佽崵濮崇欢銈囨閺囥垺鍋╃紓浣姑欢鐐存叏濮楀棗澧柣搴ゅ吹缁辨帡顢曢妶鍥х闁诲骸鐏氬畝绋款嚕?
        from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, \
            ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"[Results] Acc: {acc:.4f}, Macro-F1: {f1_macro:.4f}, Weighted-F1: {f1_weighted:.4f}")

        # 4. 闂備焦鐪归崹濠氬窗閹版澘鍨傛慨妯煎仺娴滃綊鏌熼崫鍕ゆい蹇涗憾閺屾盯寮介妸褍顫ч柣鐘冲姀閺呮稖褰侀梺绯曞墲閻熝囧矗閹扮増?(Normalized Confusion Matrix)
        # normalize='true' 闂佽崵鍋炵粙蹇涘礉鎼淬劌桅婵娉涚粻鏉款熆鐠哄搫鐦ㄩ柡?闂備焦妞挎禍鐐哄窗鎼淬劍鍋柛鏇ㄥ灠閸愨偓闂佹悶鍎弲鈺呭礉?闁荤喐绮庢晶妤冩暜婵犲嫮鍗氶柟缁㈠枛缁€宀勬煛瀹ュ洦鏆╃紒鈧径鎰€甸柣銏☆問閺€鐗堛亜椤愶綆娈橀柣銉邯瀹曞綊顢欓懖鈹垮仏?Recall
        cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

        # 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉墮閸愨偓闂佹悶鍎弲鈺呭礉椤曗偓閺屾稑顫濋鍌傘倗?
        # 闂備礁鎼粔鐑斤綖婢跺﹦?class_to_idx 闂備礁鎲￠悷銉х矓閹绢喖绠栨繝濠傜墕缁犳娊鏌熼悜妯虹仼缁炬澘绉归弻銈夊箚瑜嶅Σ鑽ゆ喐?labels
        sorted_labels = sorted(class_to_idx.items(), key=lambda x: x[1])
        # 濠电偛顕慨鎾箠韫囨拹鐔稿緞瀹€鈧惌鍡涙煕濠靛棗顏柟鐣屽█閺屾盯鏁傞崫鍕瀷濠电偛鐗婇崹褰掓偖閸撗勫珰闁圭粯甯為崝?
        unique_ids = sorted(list(set(y_true) | set(y_pred)))
        target_names = [name for name, idx in sorted_labels if idx in unique_ids]

        print("\n[Normalized Confusion Matrix] (Rows=True, Cols=Pred)")
        # 闂備胶鎳撻悘姘跺箰缁嬭法鏆ら柛鎰靛枟閸庢鏌嶉崫鍕櫤闁宠鐗撻弻鈩冩媴鐟欏嫮鍑￠梺?(濠电儑绲藉ú锔炬崲閸曨垰姹查柍?濠电偠鎻徊鐣岀矓閹绢喗鍎嶆い鎺戝?
        print(np.round(cm_norm, 2))

        # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞閹邦収姊块梺閫炲苯澧伴柟鑼閹峰懘宕ㄦ繝鍛儓闂備礁鎼悧鍡欑矓鐎涙ɑ鍙忛柣鏃傚帶闂傤垶鏌曟繛褍濂旂槐鎾寸箾閿濆懏澶勭紒璇插€块幃娲Ω閳轰胶鍊為梺鍛婃寙閸愮偓?
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=target_names)
            # values_format='.2f' 闂備礁鎼€氼剚鏅舵禒瀣︽慨姗嗗幘閳绘棃鏌曢崼婵嗩伀缂佸鎸抽幃妯跨疀閹惧墎顔夐梺?
            disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
            plt.title(f"Normalized Confusion Matrix - {dataset_name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            # 濠电儑绲藉ú锔炬崲閸岀偞鍋ら柕濞炬櫅閻愬﹪鏌涢幘妤€鍠氶弳?(闂備礁鎲￠悷顖炲垂閸洖?
            # plt.savefig(f"cm_{dataset_name}.png")
            # plt.close()
            # 婵犵數鍋涢ˇ顓㈠礉瀹ュ绀堝ù鐓庣摠閺咁剚鎱ㄥ鍡楀闂婎剦鍓熼弻锟犲礃椤忓嫅锝嗕繆椤栨凹妲虹紒顔肩墦瀹曨偊宕熼鍌氱稊闂備焦鐪归崝宀€鈧凹鍣ｉ悰顕€宕卞Ο缁樼彿闂佸憡鍓崱妤€绠戝┑鐐村灦閹稿摜绮旈悜濮愨偓鍛存晜閻愵剙顏╅梺鍛婂姦娴滄繈?plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

        # 闂佽崵濮村ù鍕繆閸モ晛鍨濋煫鍥ㄧ☉缁€鍡涙煕閳╁啰鎳嗘い顐畵閺岀喖鐓幓鎺戝?
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, labels=unique_ids, target_names=target_names, digits=4,
                                    zero_division=0))

        return {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cm_norm": cm_norm
        }

    # gpt
    def evaluate_binary(self, dataset_name: str, dataset_df: pd.DataFrame,
                        dataset_spec, class_to_idx, benign_class_idx=0,
                        do_resampled_eval=False, decision_threshold=0.5,
                        search_best_threshold=False, min_search_precision=0.90,
                        min_search_recall=0.90):
        print(f"\n--- Evaluation [{dataset_name}] (Binary) ---")

        if dataset_df.empty:
            return {}

        # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?
        self.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
                          evaluation_percent=1.0, fit_preprocessor=False)

        if self.X is None or len(self.X) == 0:
            return {}

        # 2. 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉厜缁憋綁鏌涢弴銊ュ?
        y_true_list, y_pred_prob_list = [], []
        eval_batch_size = 512
        yielder = self.get_batch_yielder(eval_batch_size, for_training=False, balanced=False)

        total_samples = len(self.X)
        total_batches_est = (total_samples + eval_batch_size - 1) // eval_batch_size
        print(f"[Eval] Predicting on {total_samples} samples (~{total_batches_est} batches)")

        batch_count = 0
        max_eval_batches = total_batches_est + 5  # 闂傚倸鍊搁崯顐㈩嚕閸洦鏁嗘繛鎴欏灩閸戠姴霉閿濆嫯顒熷ù鐓庡€块弻鐔碱敍濮橆厼娅ｉ梺鍝ュТ閺堫剛绮欐径灞稿亾閿濆骸浜介柛鏂诲劤閳ь剛鏁婚崑濠冪閻愯櫣?

        while True:
            bx, by_idx = yielder.get_batch()
            if not bx:
                print("[Eval] Reached end of evaluation batches.")
                break

            outputs = self.m.predict(bx, verbose=0)
            prob = outputs[0] if isinstance(outputs, list) else outputs

            y_true_list.extend(by_idx)
            y_pred_prob_list.extend(np.array(prob).reshape(-1))

            batch_count += 1
            if batch_count % 50 == 0 or batch_count == total_batches_est:
                print(f"[Eval] {batch_count}/{total_batches_est}")

            if batch_count > max_eval_batches:
                print("[Eval Warning] Batch count exceeded expected range, forcing stop.")
                break

        y_true = np.array(y_true_list)
        y_prob = np.array(y_pred_prob_list)

        mask = y_true != -1
        y_true, y_prob = y_true[mask], y_prob[mask]

        y_true_bin = (y_true != benign_class_idx).astype(int)
        n_benign = int(np.sum(y_true_bin == 0))
        n_attack = int(np.sum(y_true_bin == 1))
        has_both_classes = (n_benign > 0 and n_attack > 0)
        print(f"[Eval] Label distribution -> Benign: {n_benign}, Attack: {n_attack}")

        def select_threshold(y_true_calc, y_prob_calc):
            thresholds = np.concatenate([
                np.linspace(0.01, 0.99, 99),
                np.linspace(0.991, 0.999, 9)
            ])
            best_valid = None
            best_any = None

            for t in thresholds:
                y_pred_calc = (y_prob_calc > t).astype(int)
                cm_calc = confusion_matrix(y_true_calc, y_pred_calc, labels=[0, 1])
                tn_calc, fp_calc, fn_calc, tp_calc = cm_calc.ravel()
                prec_calc = precision_score(y_true_calc, y_pred_calc, zero_division=0)
                recall_calc = recall_score(y_true_calc, y_pred_calc, zero_division=0)
                f1_calc = f1_score(y_true_calc, y_pred_calc, zero_division=0)
                fpr_calc = fp_calc / (fp_calc + tn_calc) if (fp_calc + tn_calc) > 0 else 0.0
                candidate = (f1_calc, recall_calc, prec_calc, -fpr_calc, float(t))

                if best_any is None or candidate > best_any:
                    best_any = candidate
                if prec_calc >= min_search_precision and recall_calc >= min_search_recall:
                    if best_valid is None or candidate > best_valid:
                        best_valid = candidate

            selected = best_valid if best_valid is not None else best_any
            selected_threshold = selected[-1] if selected is not None else float(decision_threshold)
            selected_mode = "balanced_90p" if best_valid is not None else "best_f1_fallback"
            return selected_threshold, selected_mode

        # =========================================================
        # A. 闂備焦妞挎禍鐐哄窗鎼淬劍鍋柛鏇ㄥ灠缁€鍡涙煕閳╁喚娈樻い蹇旀尦閹綊宕惰椤徰囨煕?
        # =========================================================
        if search_best_threshold and has_both_classes:
            best_threshold_real, threshold_mode = select_threshold(y_true_bin, y_prob)
            print(
                f"[Threshold Search] Selected = {best_threshold_real:.3f} "
                f"({threshold_mode}, target Precision/Recall >= {min_search_precision:.2f}/{min_search_recall:.2f})"
            )
        else:
            best_threshold_real = float(decision_threshold)
            threshold_mode = "fixed"
        y_pred_real = (y_prob > best_threshold_real).astype(int)

        cm_real = confusion_matrix(y_true_bin, y_pred_real, labels=[0, 1])
        acc_real = accuracy_score(y_true_bin, y_pred_real)
        tn_real, fp_real, fn_real, tp_real = cm_real.ravel()
        fpr_real = fp_real / (fp_real + tn_real) if (fp_real + tn_real) > 0 else 0.0

        if has_both_classes:
            bal_acc_real = balanced_accuracy_score(y_true_bin, y_pred_real)
            recall_real = recall_score(y_true_bin, y_pred_real, zero_division=0)
            prec_real = precision_score(y_true_bin, y_pred_real, zero_division=0)
            f1_real = f1_score(y_true_bin, y_pred_real, zero_division=0)
            try:
                pr_auc_real = average_precision_score(y_true_bin, y_prob)
            except ValueError:
                pr_auc_real = np.nan
            eval_status = "ok"
        else:
            bal_acc_real = np.nan
            recall_real = np.nan
            prec_real = np.nan
            f1_real = np.nan
            pr_auc_real = np.nan
            eval_status = "single_class_ground_truth"
            print("[Eval Warning] Only one class is present in y_true; F1/Recall/Precision/BalAcc/PR-AUC are not meaningful for this split.")

        print(f"[Real Eval] Threshold = {best_threshold_real:.3f} ({threshold_mode})")
        print(
            f"[Real Eval] Acc: {acc_real:.4f}, BalAcc: {bal_acc_real:.4f}, F1: {f1_real:.4f}, Recall: {recall_real:.4f}, Precision: {prec_real:.4f}, FPR: {fpr_real:.4f}, PR-AUC: {pr_auc_real:.4f}")
        print(f"[Real Eval] CM:\n{cm_real}")

        results = {
            "eval_status": eval_status,
            "has_both_classes": has_both_classes,
            "n_benign": n_benign,
            "n_attack": n_attack,
            "real_acc": acc_real,
            "real_bal_acc": bal_acc_real,
            "real_f1": f1_real,
            "real_recall": recall_real,
            "real_prec": prec_real,
            "real_fpr": fpr_real,
            "real_pr_auc": pr_auc_real,
            "real_cm": cm_real,
            "real_best_threshold": best_threshold_real,
            "threshold_mode": threshold_mode
        }

        # =========================================================
        # B. 闂備胶顢婇惌鍥礃閵娧冨箑婵犳鍣徊楣冨蓟瑜忓▎銏ゅΨ瑜庣€氭岸鏌涢弴銊ょ盎鐎殿喚鍏橀弻銊モ槈濡厧顣洪悷婊呭閻擄繝寮婚崨顔肩窞婵炴垶姘ㄩˇ?
        # =========================================================
        if do_resampled_eval:
            idx_attack = np.where(y_true_bin == 1)[0]
            idx_benign = np.where(y_true_bin == 0)[0]

            n_attack = len(idx_attack)
            target_n_benign = min(len(idx_benign), n_attack * 10)
            target_n_benign = max(target_n_benign, 1000)
            target_n_benign = min(target_n_benign, len(idx_benign))

            if n_attack > 0:
                rng = np.random.RandomState(42)
                selected_benign_idx = rng.choice(idx_benign, size=target_n_benign, replace=False)
                keep_indices = np.concatenate([idx_attack, selected_benign_idx])

                y_true_calc = y_true_bin[keep_indices]
                y_prob_calc = y_prob[keep_indices]

                print(f"[Resampled Eval] Original: {len(idx_benign)} Benign vs {n_attack} Attack")
                print(
                    f"[Resampled Eval] Resampled: {len(selected_benign_idx)} Benign vs {n_attack} Attack (Ratio ~10:1)")
            else:
                y_true_calc = y_true_bin
                y_prob_calc = y_prob

            best_threshold_res = best_threshold_real
            y_pred_res = (y_prob_calc > best_threshold_res).astype(int)

            cm_res = confusion_matrix(y_true_calc, y_pred_res, labels=[0, 1])
            acc_res = accuracy_score(y_true_calc, y_pred_res)
            tn_res, fp_res, fn_res, tp_res = cm_res.ravel()
            fpr_res = fp_res / (fp_res + tn_res) if (fp_res + tn_res) > 0 else 0.0

            if len(np.unique(y_true_calc)) >= 2:
                bal_acc_res = balanced_accuracy_score(y_true_calc, y_pred_res)
                recall_res = recall_score(y_true_calc, y_pred_res, zero_division=0)
                prec_res = precision_score(y_true_calc, y_pred_res, zero_division=0)
                f1_res = f1_score(y_true_calc, y_pred_res, zero_division=0)
                try:
                    pr_auc_res = average_precision_score(y_true_calc, y_prob_calc)
                except ValueError:
                    pr_auc_res = np.nan
            else:
                bal_acc_res = np.nan
                recall_res = np.nan
                prec_res = np.nan
                f1_res = np.nan
                pr_auc_res = np.nan

            print(f"[Resampled Eval] Threshold = {best_threshold_res:.3f} ({threshold_mode})")
            print(
                f"[Resampled Eval] Acc: {acc_res:.4f}, BalAcc: {bal_acc_res:.4f}, F1: {f1_res:.4f}, Recall: {recall_res:.4f}, Precision: {prec_res:.4f}, FPR: {fpr_res:.4f}, PR-AUC: {pr_auc_res:.4f}")
            print(f"[Resampled Eval] CM:\n{cm_res}")

            results.update({
                "resampled_acc": acc_res,
                "resampled_bal_acc": bal_acc_res,
                "resampled_f1": f1_res,
                "resampled_recall": recall_res,
                "resampled_prec": prec_res,
                "resampled_fpr": fpr_res,
                "resampled_pr_auc": pr_auc_res,
                "resampled_cm": cm_res,
                "resampled_best_threshold": best_threshold_res
            })

        return results



    # gpt闂備礁鎲￠幐鍝ョ矓瀹曞洨鍗氶柟缁㈠枟?
    # def evaluate_binary(self, dataset_name: str, dataset_df: pd.DataFrame,
    #                     dataset_spec, class_to_idx, benign_class_idx=0,
    #                     do_resampled_eval=False):
    #
    #     print(f"\n--- Evaluation [{dataset_name}] (Binary) ---")
    #
    #     if dataset_df.empty:
    #         return {}
    #
    #     # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?
    #     self.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
    #                       evaluation_percent=1.0, fit_preprocessor=False)
    #
    #     if self.X is None or len(self.X) == 0:
    #         return {}
    #
    #     # 2. 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉厜缁憋綁鏌涢弴銊ュ?
    #     y_true_list, y_pred_prob_list = [], []
    #     yielder = self.get_batch_yielder(512, for_training=False, balanced=False)
    #
    #     while True:
    #         bx, by_idx = yielder.get_batch()
    #         if not bx:
    #             break
    #         outputs = self.m.predict(bx, verbose=0)
    #         prob = outputs[0] if isinstance(outputs, list) else outputs
    #         y_true_list.extend(by_idx)
    #         y_pred_prob_list.extend(np.array(prob).reshape(-1))
    #
    #     y_true = np.array(y_true_list)
    #     y_prob = np.array(y_pred_prob_list)
    #
    #     mask = y_true != -1
    #     y_true, y_prob = y_true[mask], y_prob[mask]
    #
    #     y_true_bin = (y_true != benign_class_idx).astype(int)
    #
    #     # =========================================================
    #     # A. 闂備焦妞挎禍鐐哄窗鎼淬劍鍋柛鏇ㄥ灠缁€鍡涙煕閳╁喚娈樻い蹇旀尦閹綊宕惰椤徰囨煕?
    #     # =========================================================
    #     thresholds = np.linspace(0.01, 0.99, 99)
    #     best_threshold_real = 0.5
    #     best_f1_real = -1.0
    #
    #     import warnings
    #     from sklearn.exceptions import UndefinedMetricWarning
    #     warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    #
    #     for t in thresholds:
    #         curr_preds = (y_prob > t).astype(int)
    #         curr_f1 = f1_score(y_true_bin, curr_preds, zero_division=0)
    #         if curr_f1 > best_f1_real:
    #             best_f1_real = curr_f1
    #             best_threshold_real = t
    #
    #     warnings.filterwarnings("default", category=UndefinedMetricWarning)
    #
    #     y_pred_real = (y_prob > best_threshold_real).astype(int)
    #
    #     cm_real = confusion_matrix(y_true_bin, y_pred_real)
    #     bal_acc_real = balanced_accuracy_score(y_true_bin, y_pred_real)
    #     recall_real = recall_score(y_true_bin, y_pred_real, zero_division=0)
    #     prec_real = precision_score(y_true_bin, y_pred_real, zero_division=0)
    #     f1_real = f1_score(y_true_bin, y_pred_real, zero_division=0)
    #
    #     try:
    #         auc_real = roc_auc_score(y_true_bin, y_prob)
    #     except ValueError:
    #         auc_real = 0.5
    #
    #     print(f"[Real Eval] Best Threshold = {best_threshold_real:.2f} (Max F1={best_f1_real:.4f})")
    #     print(
    #         f"[Real Eval] BalAcc: {bal_acc_real:.4f}, F1: {f1_real:.4f}, Recall: {recall_real:.4f}, Precision: {prec_real:.4f}, AUC: {auc_real:.4f}")
    #     print(f"[Real Eval] CM:\n{cm_real}")
    #
    #     results = {
    #         "real_bal_acc": bal_acc_real,
    #         "real_f1": f1_real,
    #         "real_recall": recall_real,
    #         "real_prec": prec_real,
    #         "real_auc": auc_real,
    #         "real_cm": cm_real,
    #         "real_best_threshold": best_threshold_real
    #     }
    #
    #     # =========================================================
    #     # B. 闂備胶顢婇惌鍥礃閵娧冨箑婵犳鍣徊楣冨蓟瑜忓▎銏ゅΨ瑜庣€氭岸鏌涢弴銊ょ盎鐎殿喚鍏橀弻銊モ槈濡厧顣洪悷婊呭閻擄繝寮婚崨顔肩窞婵炴垶姘ㄩˇ?
    #     # =========================================================
    #     if do_resampled_eval:
    #         idx_attack = np.where(y_true_bin == 1)[0]
    #         idx_benign = np.where(y_true_bin == 0)[0]
    #
    #         n_attack = len(idx_attack)
    #         target_n_benign = min(len(idx_benign), n_attack * 10)
    #         target_n_benign = max(target_n_benign, 1000)
    #         target_n_benign = min(target_n_benign, len(idx_benign))
    #
    #         if n_attack > 0:
    #             rng = np.random.RandomState(42)
    #             selected_benign_idx = rng.choice(idx_benign, size=target_n_benign, replace=False)
    #             keep_indices = np.concatenate([idx_attack, selected_benign_idx])
    #
    #             y_true_calc = y_true_bin[keep_indices]
    #             y_prob_calc = y_prob[keep_indices]
    #
    #             print(f"[Resampled Eval] Original: {len(idx_benign)} Benign vs {n_attack} Attack")
    #             print(
    #                 f"[Resampled Eval] Resampled: {len(selected_benign_idx)} Benign vs {n_attack} Attack (Ratio ~10:1)")
    #         else:
    #             y_true_calc = y_true_bin
    #             y_prob_calc = y_prob
    #
    #         best_threshold_res = 0.5
    #         best_f1_res = -1.0
    #
    #         warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    #         for t in thresholds:
    #             curr_preds = (y_prob_calc > t).astype(int)
    #             curr_f1 = f1_score(y_true_calc, curr_preds, zero_division=0)
    #             if curr_f1 > best_f1_res:
    #                 best_f1_res = curr_f1
    #                 best_threshold_res = t
    #         warnings.filterwarnings("default", category=UndefinedMetricWarning)
    #
    #         y_pred_res = (y_prob_calc > best_threshold_res).astype(int)
    #
    #         cm_res = confusion_matrix(y_true_calc, y_pred_res)
    #         bal_acc_res = balanced_accuracy_score(y_true_calc, y_pred_res)
    #         recall_res = recall_score(y_true_calc, y_pred_res, zero_division=0)
    #         prec_res = precision_score(y_true_calc, y_pred_res, zero_division=0)
    #         f1_res = f1_score(y_true_calc, y_pred_res, zero_division=0)
    #
    #         try:
    #             auc_res = roc_auc_score(y_true_calc, y_prob_calc)
    #         except ValueError:
    #             auc_res = 0.5
    #
    #         print(f"[Resampled Eval] Best Threshold = {best_threshold_res:.2f} (Max F1={best_f1_res:.4f})")
    #         print(
    #             f"[Resampled Eval] BalAcc: {bal_acc_res:.4f}, F1: {f1_res:.4f}, Recall: {recall_res:.4f}, Precision: {prec_res:.4f}, AUC: {auc_res:.4f}")
    #         print(f"[Resampled Eval] CM:\n{cm_res}")
    #
    #         results.update({
    #             "resampled_bal_acc": bal_acc_res,
    #             "resampled_f1": f1_res,
    #             "resampled_recall": recall_res,
    #             "resampled_prec": prec_res,
    #             "resampled_auc": auc_res,
    #             "resampled_cm": cm_res,
    #             "resampled_best_threshold": best_threshold_res
    #         })
    #     return results

    # def evaluate_binary(self, dataset_name: str, dataset_df: pd.DataFrame,
    #                     dataset_spec, class_to_idx, benign_class_idx=0):
    #     print(f"\n--- Evaluation [{dataset_name}] (Binary & Balanced Sampling) ---")
    #
    #     if dataset_df.empty:
    #         return {}
    #
    #     # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?
    #     self.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
    #                       evaluation_percent=1.0, fit_preprocessor=False)
    #
    #     if self.X is None or len(self.X) == 0:
    #         return {}
    #
    #     # 2. 闂備礁鍚嬮崕鎶藉床閼艰翰浜归柛銉厜缁憋綁鏌涢弴銊ュ闁靛牆顕槐鎾存媴鐟欏嫬闉嶉梺?
    #     y_true_list, y_pred_prob_list = [], []
    #     # 婵犵數鍋涢ˇ顓㈠礉瀹ュ绀堝ù鐓庣摠閺咁剚鎱ㄥΟ铏癸紞缂佺姴缍婂?Evaluation 闂傚鍋勫ú銈夊箠濮椻偓婵＄绠涘☉娆忎缓闂侀潧艌閺呮粍鎱ㄥ鍕垫闁绘劘灏欓弸鍐煏閸″繐浜炬繝纰樻閸亪鍩€椤掆偓绾绢厾娑甸埀顒勬⒒娓氬洤澧紒澶嬫綑椤斿繑绻濆顒傤吋闂侀€炲苯澧寸€殿喖鐏氬鍕偓锝庡亝椤斿秹姊虹涵鍜佸殐闁革綆鍨冲Σ鎰攽鐎ｎ亞顓洪梺鐟扮摠缁秶鍒掗敃鍌涚厱闁挎柨鎼慨鈧銈嗘尭閹芥粎鍒掔拠鍙傛梹鎷呴崫銉︹枌闂備礁鎼粔鏉懨洪妶鍡樻珷闁绘顕х粈鍐偓骞垮劚濡路娓氣偓閺屾盯寮拠鎻掝潎闂侀潻绲奸崡鍐差嚕?
    #     yielder = self.get_batch_yielder(512, for_training=False, balanced=False)
    #
    #     while True:
    #         bx, by_idx = yielder.get_batch()
    #         if not bx: break
    #         outputs = self.m.predict(bx, verbose=0)
    #         prob = outputs[0] if isinstance(outputs, list) else outputs
    #         y_true_list.extend(by_idx)
    #         y_pred_prob_list.extend(np.array(prob).reshape(-1))
    #
    #     y_true = np.array(y_true_list)
    #     y_prob = np.array(y_pred_prob_list)
    #
    #     # 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦猾宥夋煟濡寧鐝繛鍛箻閺屸€愁吋閸涱喖绐涘?
    #     mask = y_true != -1
    #     y_true, y_prob = y_true[mask], y_prob[mask]
    #
    #     # 闂佸搫顦遍崕鎰板礂濞戞碍瀚婚柣鏂挎憸椤╂煡鎮橀悙鑸殿棄闁诲氦宕电槐鎺楊敃閵堝洤绗￠梺缁樼壄缁茶法?(0: 闂備浇澹堥鏍垂闁秴? 1: 闂備浇銆€閸嬫捇鎮归崶銊ョ祷?
    #     y_true_bin = (y_true != benign_class_idx).astype(int)
    #
    #     # =========================================================
    #     # 闂備礁鎼粔鍫曗€﹂崼銏㈢处闁绘劦鍓涢埞宥嗐亜閺冨洤袚闁哄棭鍙冮弻銊モ槈濞嗗簼瀛╅梺浼欑畱鐎涒晝鈧潧鍚嬬€佃偐鈧稒顭堥崑渚€姊绘担鐟扮祷缂佸缍婂畷铏閺夋垵?(Resampling for Metric Calculation)
    #     # =========================================================
    #     # 闂備胶鍎甸弲鈺呭窗濡ゅ懏鍋夐柨婵嗩槹閺咁剚鎱ㄥΟ璇插?闂備浇澹堥鏍垂闁秴?闂備浇銆€閸嬫捇鎮归崶銊ョ祷?闂備焦鐪归崝宀€鈧凹鍙冮幆渚€顢欓悾宀€鎳濋柣搴秵閸嬪嫭寰勫澶婄骇?10:1 (闂備礁鎲￠悷銉╁磹娴犲鐒垫い鎺嗗亾闁哥啿鏅濈槐鐐差潨閳ь剟骞嗛崘顔肩妞ゆ帊鑳惰ぐ顖炴⒒娓氬洤浜濋柣顓у枤閼洪亶鍩￠崨顓狀槴闂佺硶鍓濋悷銊╋綖閿旇姤鍙忛柨婵嗘閻擄絿?
    #     # 闂佸搫顦弲婊堟偡閿曞倹?Task 0 ?F1 闂佽绻愮换鎰亹婢跺瞼绠斿璺侯儏椤曡鲸鎱ㄥ鍡楀缂佲偓閸績妲堥柟鎹愬吹缁辨澘顭块悷鐗堝枠妤犵偐鍋撻悷婊冮叄閹囨焼瀹ュ懎鐝樺銈嗗坊閸嬫捇鏌曢崶褝鏀荤€垫澘瀚～婵嬪棘閵夈儰澹曢悗鍏夊亾闁告劗鍋撻拺澶愭⒑閸涘﹤濮傞柛濠冪墵閹鈧稒顭囬々?
    #
    #     idx_attack = np.where(y_true_bin == 1)[0]
    #     idx_benign = np.where(y_true_bin == 0)[0]
    #
    #     n_attack = len(idx_attack)
    #     # 闂佽崵濮抽悞锕傚磿閹惰姤鍋╅柟顖ｇ仢閹烘妫橀柛鎾楀倻鐭楅梻浣瑰缁嬫帞鍠婂澶婃瀬濡わ絽鍟粈鍕磽娴ｈ鐒藉鐟邦樀閺岋繝宕奸銏狀潾?10 闂備胶顭堥敃銈囩矙閸曨厽顫曟繝闈涱儏缁狅綁鏌″搴″幋闁稿鎹囧畷銊╊敊閹存帞甯涘┑鐘灱閸╂牕顫濋妸鈺佹瀬闁靛牆娲犻崑鎾绘偡閻楀牊鎷辨繝鈷€鍐х€殿喖鐏氬鍕償閳藉棙顥岄梻浣筋嚃閸庢悂宕熼鐑嗘?
    #     target_n_benign = min(len(idx_benign), n_attack * 10)
    #
    #     # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞閹邦剛鍝楅柣鐘叉处瑜板啫袙婵犲洦鐓涚€广儱顦伴弫杈ㄧ節閳ь剟鎮ч崼婵堢獮濡炪倕绻愮€氼喖袙瀹€鍕叆婵炴垶顭囨晶娑㈡煟閿旇鏋熷ǎ鍥э躬瀹曟帒顫濋鍌欏摋闁诲孩顔栭崰鎺楀磻閹炬枼鏀芥い鏃傗拡閸庢粎绱掓潏銊ュ摵闁轰礁绉撮悾婵嬪礃閵婏富鍟庨梺璇茬箰缁绘垿宕瑰畷鍥ㄥ床婵鍩栭崑?1000 濠电偞鍨堕幖鈺傜閿濆棛鐝堕柛銉墮缁犳垹鎲告惔銊﹀亱闂侇剙绉寸€氬銇勯幒鎴濐仾娴滄稑鈹戦鐣岀細闁搞劏娅ｆ禍绋库攽鐎ｎ亞?
    #     target_n_benign = max(target_n_benign, 1000)
    #     target_n_benign = min(target_n_benign, len(idx_benign))
    #
    #     # 闂傚倸鍊搁幊搴ｆ崲閹存績鏀﹂柍褜鍓熼弻鐔煎箚閻楀牜妫勯悷婊勬緲濞差參骞婂┑瀣垫晬婵炴垶甯掓禍鍓ф喐鎼淬劍鍋嬮梺顒€绉?
    #     if n_attack > 0:
    #         rng = np.random.RandomState(42)
    #         selected_benign_idx = rng.choice(idx_benign, size=target_n_benign, replace=False)
    #
    #         # 闂備礁鎲￠懝楣冩偋閸℃稒鍤愰柣鏃傚劋椤洟鏌曡箛鏇炐ョ紒?
    #         keep_indices = np.concatenate([idx_attack, selected_benign_idx])
    #
    #         # 缂傚倷鐒︾粙鎺楁儎椤栫偛鐒垫い鎺嗗亾妞わ缚鍗冲鎶芥偄閻撳海顔夊銈嗘⒒閸嬫捇寮宠箛鎾佺懓顭ㄩ崘鈺佹敪濡炪倖鎸搁幗婊呭垝鐠囧弬鏃€鎷呴崫銉︹枌?
    #         y_true_calc = y_true_bin[keep_indices]
    #         y_prob_calc = y_prob[keep_indices]
    #
    #         print(f"[Eval Adjustment] Original: {len(idx_benign)} Benign vs {n_attack} Attack")
    #         print(
    #             f"[Eval Adjustment] Resampled: {len(selected_benign_idx)} Benign vs {n_attack} Attack (Ratio ~10:1)")
    #     else:
    #         y_true_calc = y_true_bin
    #         y_prob_calc = y_prob
    #
    #     # --- 闂備礁鎲￠弻锝夊礉瀹ュ鐒垫い鎴ｆ硶缁愭梹淇婇幓鎺撳枠鐎规洘鍔欏畷妤呭礂閻撳簼铏庣紓?(濠电偠鎻紞鈧繛澶嬫礋瀵偊濡舵径瀣帓閻庡箍鍎遍ˇ顖氣枍濮樿埖鐓涚€广儱鎳庡Σ濠氭煙椤旂瓔娈滈柟顖氬暣瀹曠喖顢楁担璇″晙? ---
    #     thresholds = np.linspace(0.01, 0.99, 99)
    #     best_threshold = 0.5
    #     best_f1 = -1.0
    #
    #     import warnings
    #     from sklearn.exceptions import UndefinedMetricWarning
    #     warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    #
    #     for t in thresholds:
    #         curr_preds = (y_prob_calc > t).astype(int)
    #         curr_f1 = f1_score(y_true_calc, curr_preds, zero_division=0)
    #         if curr_f1 > best_f1:
    #             best_f1 = curr_f1
    #             best_threshold = t
    #
    #     warnings.filterwarnings("default", category=UndefinedMetricWarning)
    #
    #     # print(f"[Eval Info] Best Threshold = {best_threshold:.1f} (Max F1={best_f1:.4f})")
    #     print(f"[Eval Info] Best Threshold = {best_threshold:.2f} (Max F1={best_f1:.4f})")
    #
    #     # 闂備焦鐪归崹濠氬窗閹版澘鍨傛慨妯挎硾鐎氬鏌嶈閸撴稓妲愰幒妤€閱囬柕澶樺枟濞堛劌?
    #     y_pred_calc = (y_prob_calc > best_threshold).astype(int)
    #
    #     # 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘缁犱即鏌涢妷锝呭?
    #     cm = confusion_matrix(y_true_calc, y_pred_calc)
    #     bal_acc = balanced_accuracy_score(y_true_calc, y_pred_calc)
    #     recall = recall_score(y_true_calc, y_pred_calc, zero_division=0)
    #     prec = precision_score(y_true_calc, y_pred_calc, zero_division=0)
    #     f1 = f1_score(y_true_calc, y_pred_calc, zero_division=0)
    #     try:
    #         auc = roc_auc_score(y_true_calc, y_prob_calc)
    #     except ValueError:
    #         auc = 0.5
    #
    #     print(
    #         f"[Results (Resampled)] Acc: {bal_acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {prec:.4f}, AUC: {auc:.4f}")
    #     print(f"Confusion Matrix:\n{cm}")
    #
    #     return {
    #         "bal_acc": bal_acc, "f1": f1, "recall": recall,
    #         "prec": prec, "auc": auc, "cm": cm, "best_threshold": best_threshold
    #     }


    # def evaluate_binary(self, dataset_name: str, dataset_df: pd.DataFrame,
    #                     dataset_spec, class_to_idx, benign_class_idx=0):
    #     print(f"\n--- Evaluation [{dataset_name}] (Binary) ---")
    #
    #     # 0. 婵犵妲呴崑鈧柛瀣崌閺岋紕浠︾拠鎻掑Г缂備線顤傞崣鍐ㄧ暦濡ゅ懎唯闁挎棁濮ら宥夋⒑绾拋鍤冮柛鐘愁殔鍗遍柟闂寸鐟欙箓鏌ㄩ弮鍥у姷缂佲偓鐎ｎ剛?
    #     if dataset_df.empty:
    #         print("[Eval Warning] Input DataFrame is empty!")
    #         return {}
    #
    #     # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?(evaluate 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀顒佺箾閹寸偞灏痪棰佸牚t濠碘槅鍋呭妯尖偓姘煎墴钘濋柍鍝勬噺?
    #     self.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
    #                       evaluation_percent=1.0, fit_preprocessor=False)
    #
    #     if self.X is None or len(self.X) == 0:
    #         print("[Eval Warning] Dataset Empty after loading/filtering!")
    #         return {}
    #
    #     y_true_list, y_pred_prob_list = [], []
    #     # 濠电偠鎻紞鈧繛澶嬫礋?Balanced=False闂備焦瀵х粙鎴︽儔閻撳函鑰块柨娑樺閸嬫捇宕烽鐐扮捕閻庢鍠曠划娆忣嚕娴煎瓨鍋愮€规洖娲㈤崑鐐烘煟閻樺弶澶勬繛鑼枎閳绘棃鏁冮崒姘鳖槴闂佺硶鍓濋〃鍫ヮ敂?
    #     yielder = self.get_batch_yielder(512, for_training=False, balanced=False)
    #
    #     print(f"Predicting on {len(self.X)} samples...")
    #     while True:
    #         bx, by_idx = yielder.get_batch()
    #         if not bx: break
    #
    #         # Model output: [prob, features] -> Take [0]
    #         outputs = self.m.predict(bx, verbose=0)
    #         prob = outputs[0] if isinstance(outputs, list) else outputs
    #
    #         # 缂備胶铏庨崣搴ㄥ窗閺囩姵?prob ?flat array
    #         prob = np.array(prob).reshape(-1)
    #
    #         y_true_list.extend(by_idx)
    #         y_pred_prob_list.extend(prob)
    #
    #     y_true = np.array(y_true_list)
    #     y_prob = np.array(y_pred_prob_list)
    #
    #     # 2. 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦猾宥夋煟濡寧鐝繛鍛箻閺屸€愁吋閸涱喖绐涘?
    #     mask = y_true != -1
    #     y_true, y_prob = y_true[mask], y_prob[mask]
    #
    #     if len(y_true) == 0:
    #         print("[Eval Error] No valid labels found (-1 only). Check ClassManager mapping!")
    #         return {}
    #
    #     # 3. 闂佸搫顦遍崕鎰板礂濞戞碍瀚婚柣鏂挎憸椤╂煡鎮橀悙鑸殿棄闁诲氦宕电槐鎺楊敃閵堝洤绗￠梺缁樼壄缁茶法?(0: 闂備浇澹堥鏍垂闁秴? 1: 闂備浇銆€閸嬫捇鎮归崶銊ョ祷?
    #     y_true_bin = (y_true != benign_class_idx).astype(int)
    #
    #     # --- Debug: 闂備胶鎳撻悘姘跺箰缁嬭法鏆ら柛鎰电厛濞硷繝鏌涚仦鍓х煀闁肩儤濞婇弻娑㈠箳閹寸儐妫炵紓?---
    #     idx_benign = (y_true_bin == 0)
    #     idx_attack = (y_true_bin == 1)
    #     p_benign = np.mean(y_prob[idx_benign]) if np.any(idx_benign) else 0.0
    #     p_attack = np.mean(y_prob[idx_attack]) if np.any(idx_attack) else 0.0
    #     print(f"[Debug] Probabilities -> Avg Benign: {p_benign:.4f} | Avg Attack: {p_attack:.4f}")
    #
    #     # --- 闂備礁鎼粔鍫曗€﹂崼銏㈢处闁绘劦鍓涢埞宥嗐亜閺冨洤袚闁哄棭鍙冮弻銊モ槈濞嗘帒顥濇繝娈垮枓閺呯娀鐛埀顒€霉閿濆牜娼愰柛瀣箘缁辨帗绗熼崹顔句痪濠电偞鎹侀崺鏍箯?F1 闂傚倸鍊搁崯顐︽偋閸℃稑?---
    #     thresholds = np.arange(0.1, 1.0, 0.1)  # [0.1, 0.2, ... 0.9]
    #     best_threshold = 0.5
    #     best_f1 = -1.0
    #
    #     # 濠电偞鍨堕幐鎼佹偤閵娿儺娓婚柛宀€鍋為ˉ鍫熺箾閸℃ɑ灏甸悽?sklearn ?zero_division 闂佽崵濮崑鎾绘煥閺囨浜鹃梺?
    #     import warnings
    #     from sklearn.exceptions import UndefinedMetricWarning
    #     warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    #
    #     for t in thresholds:
    #         curr_preds = (y_prob > t).astype(int)
    #         curr_f1 = f1_score(y_true_bin, curr_preds, zero_division=0)
    #         if curr_f1 > best_f1:
    #             best_f1 = curr_f1
    #             best_threshold = t
    #
    #     warnings.filterwarnings("default", category=UndefinedMetricWarning)
    #
    #     print(f"[Eval Info] Dynamic Threshold Search: Best Threshold = {best_threshold:.1f} (Max F1={best_f1:.4f})")
    #
    #     # 4. 濠电偠鎻紞鈧繛澶嬫礋瀵偊濡舵径濠傜彉闂侀€炲苯澧柟宄版嚇楠炲酣鎳為妷顔界秼闂備胶顭堥敃銊╁磻閹版澘鏋侀柟鎹愵嚙缁狅綁鏌熼柇锕€骞楀ù鐘愁焽缁辨挻鎷呴崫銉モ叺濡炪倧绲洪弲娑㈡箒?
    #     y_pred_bin = (y_prob > best_threshold).astype(int)
    #
    #     # 5. 闂備胶鎳撻悘姘跺箰缁嬭法鏆ら柛鎰靛枛閺嬩線鏌ｅΔ鈧悧鍡欑矈閿曞倹鐓曢柟鐑樺灦椤ユ粎?
    #     n_benign = np.sum(y_true_bin == 0)
    #     n_attack = np.sum(y_true_bin == 1)
    #     print(f"[Eval Info] Samples: Benign={n_benign}, Attack={n_attack}, Total={len(y_true_bin)}")
    #
    #     if n_benign == 0 or n_attack == 0:
    #         print("[Eval Warning] Test set is pure (only one class). AUC cannot be calculated correctly.")
    #
    #     # 6. 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘鐎氬鏌嶈閸撴稓妲愰幒妤€閱囬柣鏃堫棑濡嫰?
    #     cm = confusion_matrix(y_true_bin, y_pred_bin)
    #     bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    #     recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    #     prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    #     f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    #     try:
    #         auc = roc_auc_score(y_true_bin, y_prob)
    #     except ValueError:
    #         auc = 0.5
    #
    #     print(
    #         f"[Results] Acc: {bal_acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {prec:.4f}, AUC: {auc:.4f}")
    #     print(f"Confusion Matrix:\n{cm}")
    #
    #     return {
    #         "bal_acc": bal_acc, "f1": f1, "recall": recall,
    #         "prec": prec, "auc": auc, "cm": cm, "best_threshold": best_threshold
    #     }

    # def evaluate_binary(self, dataset_name: str, dataset_df: pd.DataFrame,
    #                     dataset_spec, class_to_idx, benign_class_idx=0):
    #     print(f"\n--- Evaluation [{dataset_name}] (Binary) ---")
    #
    #     # 0. 婵犵妲呴崑鈧柛瀣崌閺岋紕浠︾拠鎻掑Г缂備線顤傞崣鍐ㄧ暦濡ゅ懎唯闁挎棁濮ら宥夋⒑绾拋鍤冮柛鐘愁殔鍗遍柟闂寸鐟欙箓鏌ㄩ弮鍥у姷缂佲偓鐎ｎ剛?
    #     if dataset_df.empty:
    #         print("[Eval Warning] Input DataFrame is empty!")
    #         return {}
    #
    #     # 1. 闂備礁鎲″缁樻叏閹灐褰掑炊椤掆偓閺嬩線鏌ｅΔ鈧悧鍡欑矈?(evaluate 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀顒佺箾閹寸偞灏痪棰佸牚t濠碘槅鍋呭妯尖偓姘煎墴钘濋柍鍝勬噺?
    #     self.load_dataset(dataset_name, dataset_df, dataset_spec, class_to_idx,
    #                       evaluation_percent=1.0, fit_preprocessor=False)
    #
    #     if self.X is None or len(self.X) == 0:
    #         print("[Eval Warning] Dataset Empty after loading/filtering!")
    #         return {}
    #
    #     y_true_list, y_pred_prob_list = [], []
    #     # 濠电偠鎻紞鈧繛澶嬫礋?Balanced=False闂備焦瀵х粙鎴︽儔閻撳函鑰块柨娑樺閸嬫捇宕烽鐐扮捕閻庢鍠曠划娆忣嚕娴煎瓨鍋愮€规洖娲㈤崑鐐烘煟閻樺弶澶勬繛鑼枎閳绘棃鏁冮崒姘鳖槴闂佺硶鍓濋〃鍫ヮ敂?
    #     yielder = self.get_batch_yielder(512, for_training=False, balanced=False)
    #
    #     print(f"Predicting on {len(self.X)} samples...")
    #     while True:
    #         bx, by_idx = yielder.get_batch()
    #         if not bx: break
    #
    #         # Model output: [prob, features] -> Take [0]
    #         outputs = self.m.predict(bx, verbose=0)
    #         prob = outputs[0] if isinstance(outputs, list) else outputs
    #
    #         # 缂備胶铏庨崣搴ㄥ窗閺囩姵?prob ?flat array
    #         prob = np.array(prob).reshape(-1)
    #
    #         y_true_list.extend(by_idx)
    #         y_pred_prob_list.extend(prob)
    #
    #     y_true = np.array(y_true_list)
    #     y_prob = np.array(y_pred_prob_list)
    #
    #     # 2. 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦猾宥夋煟濡寧鐝繛鍛箻閺屸€愁吋閸涱喖绐涘?
    #     mask = y_true != -1
    #     y_true, y_prob = y_true[mask], y_prob[mask]
    #
    #     # --- 闂備礁鎼崐鐟邦熆濮椻偓?Debug 濠电偛顕刊瀵稿緤閸ф?---
    #     avg_prob_benign = np.mean(y_prob[y_true == benign_class_idx])
    #     avg_prob_attack = np.mean(y_prob[y_true != benign_class_idx])
    #     print(f"[Debug] Avg Prob Benign: {avg_prob_benign:.4f}, Avg Prob Attack: {avg_prob_attack:.4f}")
    #     # ---------------------
    #
    #     if len(y_true) == 0:
    #         print("[Eval Error] No valid labels found (-1 only). Check ClassManager mapping!")
    #         return {}
    #
    #     # 3. 闂佸搫顦遍崕鎰板礂濞戞碍瀚婚柣鏂挎憸椤╂煡鎮橀悙鑸殿棄闁诲氦宕电槐鎺楊敃閵堝洤绗￠梺缁樼壄缁茶法?
    #     y_true_bin = (y_true != benign_class_idx).astype(int)
    #     y_pred_bin = (y_prob > 0.5).astype(int)  # 闂傚倸鍊搁崯顐︽偋閸℃稑?0.5
    #
    #     # 4. 闂備胶鎳撻悘姘跺箰缁嬭法鏆ら柛鎰靛枛閺嬩線鏌ｅΔ鈧悧鍡欑矈閿曞倹鐓曢柟鐑樺灦椤ユ粎?(Debug)
    #     n_benign = np.sum(y_true_bin == 0)
    #     n_attack = np.sum(y_true_bin == 1)
    #     print(f"[Eval Info] Samples: Benign={n_benign}, Attack={n_attack}, Total={len(y_true_bin)}")
    #
    #     if n_benign == 0 or n_attack == 0:
    #         print("[Eval Warning] Test set is pure (only one class). AUC cannot be calculated correctly.")
    #
    #     # 5. 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘缁犱即鏌涢妷锝呭?
    #     cm = confusion_matrix(y_true_bin, y_pred_bin)
    #     bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    #     recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    #     prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    #     f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    #     try:
    #         auc = roc_auc_score(y_true_bin, y_prob)
    #     except ValueError:
    #         auc = 0.5  # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞閹邦剛鐓戝銈呯箰閹冲繒绮婚幒鎴旀闁圭虎鍨版禍鐐箾閹寸偞鎯勫ù婊呭仧閹噣顢曢敃鈧粈鍡樸亜閹伴潧澧扮紒鈧崼绺怌 闂備礁鎼崯鐗堟叏閻㈠灚鍏滈柛鎾茶閸嬫挾鎲撮崟顓犲彎?
    #
    #     print(f"[Results] Acc: {bal_acc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")
    #     print(f"Confusion Matrix:\n{cm}")
    #
    #     return {
    #         "bal_acc": bal_acc, "f1": f1, "recall": recall,
    #         "prec": prec, "auc": auc, "cm": cm
    #     }

    # ---------------------------------------------------------------
    # Utility Functions (Keeping original logic for loading)
    # ---------------------------------------------------------------
    def load_dataset(self, dataset_name: str, dataset: Union[pd.DataFrame, str], specification: DatasetSpecification,
                     class_to_idx: Dict[str, int], cache_path: Optional[str] = None, n_rows: int = 0,
                     evaluation_dataset_sampling: EvaluationDatasetSampling = EvaluationDatasetSampling.LastRows,
                     evaluation_percent: float = 0.2, numerical_filter=1_000_000_000,
                     fit_preprocessor: bool = True) -> pd.DataFrame:
        if cache_path is not None and not os.path.exists(cache_path): os.mkdir(cache_path)
        self.dataset_specification = specification
        self.class_to_idx = class_to_idx
        df, model_input_spec = self._load_preprocessed_dataset(dataset_name, dataset, specification, cache_path, n_rows,
                                                               evaluation_dataset_sampling, evaluation_percent,
                                                               numerical_filter, fit_preprocessor=fit_preprocessor)
        self.training_mask = df["__training"].values;
        del df["__training"]
        self.y = pd.Series(df["__y"].values).map(class_to_idx).fillna(-1).astype(int);
        del df["__y"]
        self.X = df;
        self.model_input_spec = model_input_spec
        X_np = self.X.values.astype(np.float32)
        # 濠电姭鎷冮崨顓濈捕?Padding ?(?)
        self.X_np_with_padding = np.vstack([X_np, np.zeros((1, X_np.shape[1]), dtype=np.float32)])
        self.X_padding_index = X_np.shape[0]
        return df

    def _load_preprocessed_dataset(self, dataset_name: str, dataset: Union[pd.DataFrame, str],
                                   specification: DatasetSpecification, cache_folder: Optional[str] = None,
                                   n_rows: int = 0,
                                   evaluation_dataset_sampling: EvaluationDatasetSampling = EvaluationDatasetSampling.LastRows,
                                   evaluation_percent: float = 0.2, numerical_filter=1_000_000_000,
                                   fit_preprocessor: bool = True) -> Tuple[pd.DataFrame, ModelInputSpecification]:
        if isinstance(dataset, str) and dataset.lower().endswith(".csv"):
            dataset = pd.read_csv(dataset, nrows=n_rows if n_rows > 0 else None)
        training_mask = np.ones(len(dataset), dtype=bool)
        eval_n = int(len(dataset) * evaluation_percent)
        if eval_n > 0 and evaluation_dataset_sampling == EvaluationDatasetSampling.LastRows: training_mask[
                                                                                             -eval_n:] = False
        numerical_columns = set(specification.include_fields).difference(specification.categorical_fields)
        categorical_columns = specification.categorical_fields
        new_df = {"__training": training_mask, "__y": dataset[specification.class_column].values}
        new_features = []

        for col_name in numerical_columns:
            new_features.append(col_name)
            col_values = dataset[col_name].values
            col_values[~np.isfinite(col_values)] = 0
            col_values[col_values < -numerical_filter] = 0
            col_values[col_values > numerical_filter] = 0
            col_values = np.log1p(np.abs(col_values))
            new_df[col_name] = col_values.astype("float32").ravel()

        for i, col_name in enumerate(numerical_columns):
            all_data = new_df[col_name]
            training_data = all_data[training_mask]
            if fit_preprocessor and len(training_data) > 0: self.pre_processing.fit_numerical(col_name, training_data)
            transformed = self.pre_processing.transform_numerical(col_name, all_data)
            if hasattr(transformed, "ndim") and transformed.ndim > 1: transformed = transformed.ravel()
            new_df[col_name] = transformed

        levels_per_categorical_feature = []
        for i, col_name in enumerate(categorical_columns):
            new_features.append(col_name)
            if col_name == specification.class_column: continue
            all_data = dataset[col_name].astype(str).values
            training_data = all_data[training_mask]
            if fit_preprocessor and len(training_data) > 0: self.pre_processing.fit_categorical(col_name, training_data)
            new_values = self.pre_processing.transform_categorical(col_name, all_data,
                                                                   self.input_encoding.required_input_format)
            if self.input_encoding.required_input_format == CategoricalFormat.OneHot:
                if isinstance(new_values, pd.DataFrame):
                    expected_cols = self.pre_processing.encoded_levels.get(col_name, [])
                    if expected_cols is not None and len(expected_cols) > 0: new_values = new_values.reindex(
                        columns=expected_cols, fill_value=0)
                    levels_per_categorical_feature.append(len(new_values.columns))
                    for c in new_values.columns: new_df[f"{col_name}_{c}"] = new_values[c].values.astype(np.float32)
                else:
                    n_one_hot_levels = new_values.shape[1]
                    levels_per_categorical_feature.append(n_one_hot_levels)
                    for z in range(n_one_hot_levels): new_df[f"{col_name}_{z}"] = new_values[:, z].astype(np.float32)
            else:
                if hasattr(new_values, "ndim") and new_values.ndim > 1: new_values = new_values.ravel()
                levels_per_categorical_feature.append(len(np.unique(new_values)))
                new_df[col_name] = new_values

        new_df = pd.DataFrame(new_df)
        model_input_spec = ModelInputSpecification(new_features, len(numerical_columns), levels_per_categorical_feature,
                                                   self.input_encoding.required_input_format)
        return new_df, model_input_spec

        # (濠电儑绲藉ù鍌炲窗濡ゅ懎?BatchYielder 缂傚倷绶￠崑澶愵敊婵犲洦鍤堥柟杈鹃檮?get_batch 闂備礁鎼崐浠嬶綖婢跺本?

    def get_batch_yielder(self, batch_size: int, for_training: bool = True, balanced: bool = True, **kwargs):
        class BatchYielder:
            def __init__(self, ft, batch_size, for_training, balanced):
                self.ft = ft
                self.batch_size = batch_size
                self.for_training = for_training
                self.balanced = balanced
                self.base_mask = self.ft.training_mask if for_training else ~self.ft.training_mask

                self.selectable_mask = np.zeros(len(self.ft.y), dtype=bool)
                window_size = self.ft.parameters.window_size
                if len(self.selectable_mask) > (2 * window_size):
                    self.selectable_mask[window_size:-window_size] = True
                else:
                    self.selectable_mask[:] = True

                self.ordered = getattr(self.ft.parameters, "_train_ensure_flows_are_ordered_within_windows", True)
                self.random = not getattr(self.ft.parameters, "_train_draw_sequential_windows", False)

                if for_training and balanced:
                    valid_mask = self.base_mask & self.selectable_mask & (self.ft.y.values != -1)
                    benign_mask = (self.ft.y.values == 0)
                    self.malicious_indices = np.argwhere(valid_mask & ~benign_mask).reshape(-1)
                    self.legit_indices = np.argwhere(valid_mask & benign_mask).reshape(-1)
                    self.cursor_malicious = 0
                    self.cursor_legit = 0
                else:
                    self.available_indices = np.where(self.base_mask)[0]
                    if for_training:
                        self.available_indices = self.available_indices[self.selectable_mask[self.available_indices]]
                    if self.available_indices.size > 0:
                        self.ft.rs.shuffle(self.available_indices)
                    self.current_pos = 0

            def _build_windows(self, indices: np.ndarray):
                if self.ordered or not self.for_training:
                    return self.ft.get_windows_for_indices(indices)

                train_indices = np.argwhere(self.ft.training_mask).reshape(-1)
                if train_indices.size == 0:
                    return self.ft.get_windows_for_indices(indices)

                context_indices_batch = self.ft.rs.choice(
                    train_indices,
                    size=(len(indices), self.ft.parameters.window_size),
                    replace=True
                )
                context_indices_batch[:, -1] = indices
                return [self.ft.X.iloc[index_row] for index_row in context_indices_batch]

            def _get_batch_unbalanced(self):
                if self.current_pos >= len(self.available_indices):
                    return [], np.array([])
                end_pos = self.current_pos + self.batch_size
                final_indices = self.available_indices[self.current_pos:end_pos]
                self.current_pos = end_pos
                windows = self._build_windows(final_indices)
                return self.ft.samplewise_to_featurewise(windows), self.ft.y.values[final_indices]

            def get_batch(self):
                if self.for_training and self.balanced:
                    if len(self.malicious_indices) == 0 or len(self.legit_indices) == 0:
                        return self._get_batch_unbalanced()

                    n_malicious = self.batch_size // 2
                    n_legit = self.batch_size - n_malicious

                    if self.random:
                        malicious_indices_batch = self.ft.rs.choice(
                            self.malicious_indices,
                            size=n_malicious,
                            replace=(len(self.malicious_indices) < n_malicious)
                        )
                        legitimate_indices_batch = self.ft.rs.choice(
                            self.legit_indices,
                            size=n_legit,
                            replace=(len(self.legit_indices) < n_legit)
                        )
                    else:
                        malicious_indices_batch = self.malicious_indices[
                            self.cursor_malicious:self.cursor_malicious + n_malicious
                        ]
                        legitimate_indices_batch = self.legit_indices[
                            self.cursor_legit:self.cursor_legit + n_legit
                        ]
                        self.cursor_malicious = (self.cursor_malicious + n_malicious) % max(
                            1, len(self.malicious_indices) - n_malicious + 1
                        )
                        self.cursor_legit = (self.cursor_legit + n_legit) % max(
                            1, len(self.legit_indices) - n_legit + 1
                        )

                    final_indices = np.concatenate([malicious_indices_batch, legitimate_indices_batch])
                    self.ft.rs.shuffle(final_indices)
                    windows = self._build_windows(final_indices)
                    return self.ft.samplewise_to_featurewise(windows), self.ft.y.values[final_indices]

                return self._get_batch_unbalanced()

        return BatchYielder(self, batch_size, for_training, balanced)


    # def get_batch_yielder(self, batch_size: int, for_training: bool = True, balanced: bool = True, **kwargs):
    #     class BatchYielder:
    #         def __init__(self, ft, batch_size, for_training, balanced):
    #             self.ft = ft;
    #             self.batch_size = batch_size;
    #             self.for_training = for_training;
    #             self.balanced = balanced
    #             self.base_mask = self.ft.training_mask if for_training else ~self.ft.training_mask
    #             self.available_indices = np.where(self.base_mask)[0]
    #             self.current_pos = 0
    #             if for_training and balanced:
    #                 y_train = self.ft.y[self.base_mask]
    #                 unique_y = np.unique(y_train)
    #                 # 闂佸搫顦弲娑樏洪敃鍌氱闁靛牆顦粻?-1 (闂備礁鎼悧婊勭濠靛柈锝堫槾濞存粍鎮傚畷鐔碱敇閳藉棙钑夐梻?
    #                 unique_y = unique_y[unique_y != -1]
    #                 self.indices_per_class = {c: np.where((self.ft.y.values == c) & self.base_mask)[0] for c in
    #                                           unique_y}
    #                 self.classes = list(self.indices_per_class.keys())
    #                 # 濠电姷顣介埀顒€鍟块埀顒€缍婇幃妯诲緞婵炵偓鐓㈤梺鏂ユ櫅閸燁垳绮婚幒鎳ㄥ綊鎮╂笟顖氭暯缂傚倸绉寸粔鐟邦嚕閸偄绶為悗锝庡亝閻濓紕绱撴笟鍥т函妞ゆ泦鍥ф槬闁糕剝绋掗弲顒傗偓鍏夊亾闁告洦鍎烽埡鍌涘弿婵犻潧瀚悘鈺冪磼鐏炵晫浠㈤柍?
    #                 if not self.classes:
    #                     self.balanced = False
    #             else:
    #                 if self.available_indices.size > 0: self.ft.rs.shuffle(self.available_indices)
    #
    #         def _get_batch_unbalanced(self):
    #             if self.current_pos >= len(self.available_indices): return [], np.array([])
    #             end_pos = self.current_pos + self.batch_size
    #             final_indices = self.available_indices[self.current_pos:end_pos]
    #             self.current_pos = end_pos
    #             return self.ft.samplewise_to_featurewise(self.ft.get_windows_for_indices(final_indices)), \
    #                    self.ft.y.values[final_indices]
    #
    #         def get_batch(self):
    #             if self.for_training and self.balanced and self.classes:
    #                 n_per_class = max(1, self.batch_size // len(self.classes))
    #                 final_indices = []
    #                 for c in self.classes:
    #                     indices = self.indices_per_class[c]
    #                     if len(indices) == 0: continue
    #                     # 闂傚倸鍊搁幊搴ｆ崲閹存績鏀﹂柍褜鍓熷娲箵閹烘挸顦╅梺缁橆殜娴滃爼寮鍥︽勃闁伙絽鏈鏍⒑缂佹鐭嬫繝銏∶…鍥ㄦ償閿濆棗?
    #                     selected = self.ft.rs.choice(indices, size=n_per_class, replace=True)
    #                     final_indices.extend(selected)
    #
    #                 final_indices = np.array(final_indices)
    #                 if len(final_indices) == 0: return [], np.array([])
    #
    #                 self.ft.rs.shuffle(final_indices)
    #                 return self.ft.samplewise_to_featurewise(self.ft.get_windows_for_indices(final_indices)), \
    #                        self.ft.y.values[final_indices]
    #             else:
    #                 return self._get_batch_unbalanced()
    #
    #     return BatchYielder(self, batch_size, for_training, balanced)

    def get_windows_for_indices(self, indices: np.ndarray) -> List[pd.DataFrame]:
        W = self.parameters.window_size
        window_offsets = np.arange(-W + 1, 1)
        indices = indices.astype(int)
        all_indices = indices[:, None] + window_offsets
        # Clip闂傚倸鍊搁崯顐㈩嚕閸洦鏁嗘繛鎴炵閹儱鈹戦悩鎻掝仾?
        all_indices = np.clip(all_indices, 0, self.X_padding_index - 1)

        # 濠电偠鎻紞鈧繛澶嬫礋?numpy 濠德板€曢崐褰掆€﹂崶顭戞闁搞儮鏅滈～鏇㈡煏韫囨洖啸缂佸苯娼￠悡顐﹀炊鐠鸿桨鍠婇梺閫炲苯澧伴柛鐘冲姌閵囨劖绗熼埀顒€?
        windows_flat = self.X_np_with_padding[all_indices.ravel()]
        windows_np = windows_flat.reshape(len(indices), W, self.X_np_with_padding.shape[1])
        return [pd.DataFrame(win, columns=self.X.columns) for win in windows_np]

    def samplewise_to_featurewise(self, X: List[pd.DataFrame]):
        if not X: return []
        W, spec = self.parameters.window_size, self.model_input_spec
        combined = pd.concat(X, ignore_index=True)
        feature_map = {}
        for f in spec.feature_names:
            if f in spec.numeric_feature_names or spec.categorical_format == CategoricalFormat.Integers:
                feature_map[f] = [f]
            else:
                feature_map[f] = [c for c in X[0].columns if str(c).startswith(f + "_")]
        featurewise_X = []
        for feat in spec.feature_names:
            if feat in feature_map and feature_map[feat]:
                valid_cols = [c for c in feature_map[feat] if c in combined.columns]
                if not valid_cols:
                    vals = np.zeros((len(combined), 1), dtype=np.float32)
                else:
                    vals = np.nan_to_num(combined[valid_cols].values.astype(np.float32))
                vals = vals.astype(np.float32)
                vals = np.array([vals[i: i + W] for i in range(0, len(vals), W)], dtype=np.float32)
                if feat in spec.numeric_feature_names and vals.ndim == 2: vals = np.expand_dims(vals, axis=-1)
                featurewise_X.append(vals)
            else:
                shape_dim = 1;
                zeros = np.zeros((len(X), W, shape_dim), dtype=np.float32);
                featurewise_X.append(zeros)
        return featurewise_X


