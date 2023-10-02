def main():
  import jax
  import jax.numpy as jnp
  import warnings
  import dreamerv3
  from dreamerv3 import embodied

  import datetime

  # import envs
  import json
  import sys
  import os

  current_dir = os.path.dirname(os.path.abspath(__file__))
  env_path = os.path.join(current_dir,'..','fmenv')
  sys.path.append(env_path)
  import env as envs

  rbd_path = os.path.join(current_dir,'..','DRA')
  sys.path.append(rbd_path)
  import net
  from flax.training.train_state import TrainState
  from flax.training import checkpoints
  import optax

  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  logname = datetime.datetime.now().strftime('run%Y%m%dT%H%M')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'logdir': '~/logdir/'+logname,
      'run.train_ratio': 4, # 64
      'run.log_every': 30,  # Seconds
      'batch_size': 2, # 16
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^', # check docs
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image', # image and ref
      'decoder.cnn_keys': 'image', # image only
      #'wrapper.length': 100,
      #'jax.platform': 'cpu',
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
  config_denoiser_path = '/home/maxbarysevich/fmenv/config.json'
  with open(config_denoiser_path,'r') as c:
    config_denoiser = json.load(c)
  init_rng = jax.random.PRNGKey(0)
  module = net.ResNet(num_blocks=3,channels=32)
  params = module.init(init_rng,
                      jnp.ones((1,
                                config_denoiser['img_dim'],
                                config_denoiser['img_dim'],
                                1))
                      )['params']
  tx = optax.adamw(0.)
  state = TrainState.create(apply_fn=module.apply,params=params,tx=tx)
  state = checkpoints.restore_checkpoint(config_denoiser['denoiser'],state,prefix='ckpt_')

  @jax.jit
  def denoiser(x):
      ref = jnp.expand_dims(x[0],axis=0)
      imn = jnp.expand_dims(x[1],axis=0)
      y = state.apply_fn({'params': state.params},[ref,imn])[0]
      return y

  #import crafter
  from embodied.envs import from_gym
  #cpu = jax.devices('cpu')[0]
  #with jax.default_device(cpu):
  env = envs.DenoiserEnv(config_denoiser_path,denoiser)  # Replace this with your Gym env.
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
