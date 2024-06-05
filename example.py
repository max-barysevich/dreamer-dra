def main():
  import os
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  import shutil

  import jax
  import jax.numpy as jnp
  import warnings
  import dreamerv3
  from dreamerv3 import embodied

  import datetime

  # import envs
  import json
  import sys

  with open('config.json','r') as c:
    config_main = json.load(c)
  
  env_path = config_main['imports']['env']
  sys.path.append(env_path)
  import env as envs

  rbd_path = config_main['imports']['net']
  sys.path.append(rbd_path)
  import net

  from flax.training.train_state import TrainState
  from flax.training import checkpoints
  import optax

  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  if config_main['training']['resume'] is None:
      logname = datetime.datetime.now().strftime('run%Y%m%dT%H%M')
  else:
      logname = config_main['training']['resume']

  if not os.path.exists(config_main['training']['logdir']+logname):
      os.mkdir(config_main['training']['logdir']+logname)

  shutil.copy('config.json',config_main['training']['logdir']+logname+f'/config_{logname}.json')
  shutil.copy(env_path+'config.json',config_main['training']['logdir']+logname+f'/env_config_{logname}.json')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'logdir': config_main['training']['logdir']+logname,
      'run.train_ratio': config_main['training']['train_ratio'], # 64
      'run.log_every': config_main['training']['log_every'],  # Seconds

      'run.steps': config_main['training']['steps'], # 1e6
      'envs.amount': config_main['training']['envs'], # 1
      'batch_size': config_main['training']['batch_size'], # 16
      'batch_length': config_main['training']['batch_length'], # 64

      'jax.prealloc': False,
      'jax.precision': 'bfloat16',

      'imag_horizon': config_main['training']['imag_horizon'], # 15

      'encoder.mlp_keys': '$^', # check docs
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': ['obs','ref'], # image and ref
      'decoder.cnn_keys': 'obs', # image only
      'encoder.cnn_blocks': config_main['training']['enc_cnn_blocks'],
      'decoder.cnn_blocks': config_main['training']['dec_cnn_blocks'],
      #'wrapper.length': 100,
      #'jax.platform': 'cpu',
      'model_opt.lr': config_main['training']['lr'],
      'loss_scales.image': config_main['training']['loss_sca_img'],
      'loss_scales.vector': config_main['training']['loss_sca_vec'],
      'loss_scales.reward': config_main['training']['loss_sca_rew'],
      'loss_scales.cont': config_main['training']['loss_sca_cont'],
      'loss_scales.dyn': config_main['training']['loss_sca_dyn'],
      'loss_scales.rep': config_main['training']['loss_sca_rep']
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  # create denoiser

  def create_denoiser_state(config,rng):
     module = net.Uformer(img_dim=config['img_dim'],
                         img_ch=1,
                         out_ch=1,
                         proj_dim=config['denoiser']['proj_dim'],
                         proj_kernel=3,
                         patch_dim=config['denoiser']['patch_dim'],
                         attn_heads=config['denoiser']['attn_heads'],
                         attn_dim=config['denoiser']['attn_dim'],
                         dropout_rate=config['denoiser']['dropout_rate'],
                         leff_filters=config['denoiser']['leff_filters'],
                         blocks=config['denoiser']['blocks'])
     variables = module.init(rng,
                            (jnp.ones((1,config['img_dim'],config['img_dim'],1)),
                             jnp.ones((1,config['img_dim'],config['img_dim'],1)))
                            )
     return module, variables

  config_denoiser_path = env_path + 'config.json'
  with open(config_denoiser_path,'r') as c:
    config_denoiser = json.load(c)
  init_rng = jax.random.PRNGKey(0)
  denoiser_module, variables = create_denoiser_state(config_denoiser,init_rng)
  state = checkpoints.restore_checkpoint(config_denoiser['denoiser']['path'],
                                         target=None)

  @jax.jit
  def denoiser(x):
      ref = jnp.expand_dims(x[0],axis=0)
      imn = jnp.expand_dims(x[1],axis=0)
      y = denoiser_module.apply({'params':state['params'],
                                 'rpi':variables['rpi'],
                                 'shift_attn_mask':variables['shift_attn_mask']},
                                 [ref,imn])[0]
      return y

  #import crafter
  from embodied.envs import from_gym
  #cpu = jax.devices('cpu')[0]
  #with jax.default_device(cpu):
  #env = envs.DenoiserEnv(config_denoiser_path,denoiser)  # Replace this with your Gym env.
  env = envs.DenoiserSparseContISERewardEnv(config_denoiser_path,denoiser)
  #env = envs.RBDEnv(config_denoiser_path)
  env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  with jax.transfer_guard("allow"):
    embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
