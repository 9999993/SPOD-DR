"""Regression for the omitted current Swin FP10 ablation release."""
import ast,contextlib,csv,hashlib,json,sys,types,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src/scripts"))
import release_predict as predict
import training_pipeline as pipeline
import training_resources as resources
VARIANTS=("NO_DESTYLE","NO_LOCAL","NO_SAM")
class SwinReleaseTests(unittest.TestCase):
 def test_three_swin_ablation_predict_specs(self):
  for variant in VARIANTS:
   with self.subTest(variant=variant):
    spec=predict.model_spec("ablation","SwinTiny",variant)
    self.assertEqual(spec["artifact_id"],"ablation/SwinTiny__"+variant)
 def test_three_swin_training_plans(self):
  for variant in VARIANTS:
   with self.subTest(variant=variant):
    plan=pipeline.task_plan("ablation","SwinTiny",variant,None,None)
    self.assertEqual(plan["module_dir"],"ablation")
    self.assertEqual(plan["env"]["SPOD_VARIANT"],variant)
 def test_swin_ablation_resource_scope(self):
  result=resources.inspect_resources(backbone="SwinTiny",family="ablation")
  self.assertEqual(result["backbones"],["SwinTiny"])
 def test_swin_sensitivity_stays_unregistered(self):
  for family in ("sensitivity",):
   with self.subTest(family=family),self.assertRaises(ValueError):predict.model_spec(family,"SwinTiny",rho=.05 if family=="sensitivity" else None)
 def test_swin_assets_and_policy_are_registered(self):
  rows=list(csv.DictReader((ROOT/"checkpoints/models.csv").open(encoding="utf-8-sig")))
  self.assertEqual(len(rows),57)
  for variant in VARIANTS:
   spec=predict.model_spec("ablation","SwinTiny",variant)
   row=predict.index_row(spec)
   weight,policy,wh,ph,tau=predict.selected_assets("ablation","SwinTiny",variant)
   self.assertEqual(json.loads(policy.read_text())["model_sha256"],wh)
   self.assertTrue((ROOT/row["config_file"]).is_file())
 def test_swin_global_only_route_uses_channel_first_features(self):
  import torch
  source=ROOT/"src/scripts/ablation/structural.py"
  node=next(x for x in ast.parse(source.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=="forward_global")
  def checked(x):
   self.assertEqual(tuple(x.shape),(1,8,2,3));return x
  axis=types.SimpleNamespace(cumulative=checked)
  base=types.SimpleNamespace(global_branch=checked,global_head=lambda z:(torch.ones(1),torch.ones(1)))
  m=types.SimpleNamespace(b="SwinTiny",backbone=object(),parent_new=False,phase=2,inner=types.SimpleNamespace(base=base,destylize_stage4=checked,axis_stage4=axis))
  pm=types.SimpleNamespace(split_prefix=lambda b,bb,x:x,split_middle=lambda b,bb,x:x,split_tail=lambda b,bb,x:x)
  ns={"contextlib":contextlib,"torch":torch,"pm":pm,"gaussian_logp":lambda mu,sigma:torch.zeros(1,5)}
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),ns)
  output=ns["forward_global"](m,torch.zeros(1,2,3,8))
  self.assertEqual(tuple(output["logp"].shape),(1,5))
if __name__=="__main__":unittest.main()
