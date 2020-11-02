# coding=utf-8
""" """

import tensorflow as tf

hparams = tf.contrib.training.HParams(
    acoustic_label_dim=626,
    dur_label_dim=617,
    acoustic_dim=75,
    dur_dim=5,

    # spss_label_dim=626, # spss_label_dim
    # spss_cmp_dim=75, # spss_label_dim
    # spss_dur_cmp_dim=5,
    # spss_dur_label_dim=617,

    batch_size=32,
    reg_weight=1e-7,  # regularization weight (for l2 regularization)
    use_regularization=True,
    random_seed=191023,
    start_decay=25000,  # Step at which learning decay starts
    decay_steps=25000,  # Decay steps
    decay_rate=0.5,  # Learning rate decay rate
    initial_learning_rate=2e-4,  # Starting learning rate
    final_learning_rate=2e-6,  # Minimal learning rate
    adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    adam_epsilon=1e-6,  # AdamOptimizer beta3 parameter
    gradclip_value=1.0,  # Gradient clipped values
    max_abs_value=4.,
    total_training_steps=200000,  # Maximum training steps
    save_summary_steps=500,  # Steps between running summary ops
    save_checkpoints_steps=2000,  # Steps between writing checkpoints
    keep_checkpoint_max=20,  # Maximum keeped model
    valid_interval=5000,  # Steps between eval on validation data

)
