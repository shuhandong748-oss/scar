import csv,json,hashlib
from pathlib import Path
R=Path(r'<SCAR_ROOT>'); O=R/'CORRECTED_RESULT_TABLE_20261104'
def read(p): return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
base=read(R/'RESULTS_UNIFIED_AUDIT_20261103/RESULTS_CHECK.csv'); old={r['id']:r for r in read(R/'SCAR_B_ALL_METHODS_20261030/TABLES/MAIN_TABLE.csv')}
sources=[R/'RESULTS_UNIFIED_AUDIT_20261103/RESULTS_CHECK.csv',R/'SCAR_B_ALL_METHODS_20261030/TABLES/MAIN_TABLE.csv']
rows=[]
for a in base:
 if a['cell_id']=='D4P_KITTI_OLD':continue
 k=a['cell_id']; x={'id':k,'method':a['method'].replace('Des4Pos-paper-reimpl','Des4Pos corrected'),'dataset':a['dataset'],'split':'TEST','unit':'percent','notes':'','source':str(sources[0])}
 for met,suf in [('top1','T1_5m'),('cell','cell_R1')]:
  for name,pre in [('Original','original'),('Presence','presence'),('Rank Fusion','rankfusion'),('FSR','fsr')]:x[name+'_'+met]=a[pre+'_'+suf]
  x['Historical A_'+met]=old[k]['A_'+('top1' if met=='top1' else 'cell')] if k in old else a['scar_'+suf]
  x['SCAR_'+met]=old[k]['B_'+('top1' if met=='top1' else 'cell')] if k in old else 'PENDING'
 if k.startswith('D4P_'):
  for m in ['top1','cell']:
   x['Historical A_'+m]='PENDING'
   if k=='D4P_KITTI_CORRECTED':x['SCAR_'+m]=a['scar_'+('T1_5m' if m=='top1' else 'cell_R1')]
   else:
    for n in ['Original','Presence','Rank Fusion','FSR','SCAR']:x[n+'_'+m]='PENDING'
  x['notes']='修正版；不使用旧重实现TEST补值。历史A在本版未开展。' if k!='D4P_KITTI_CORRECTED' else 'test13修正版追加探索性TEST；未开展同主干历史A选轮对照。'
 if k in ('D4P_NPM3D','D4P_S3DIS'):
  domain=k.split('_')[1].lower()
  delivery=R/'D4P_CORRECTED_TEST_FOUR_DOMAINS_20261104'/domain
  manifest={name.strip():sha for sha,name in (line.split(maxsplit=1) for line in (delivery/'SHA256SUMS_DELIVERY.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip())}
  for name in ('eval/metrics.json','VERIFY_TEST.json'):
   path=delivery/name
   assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[name],path
   sources.append(path)
  assert json.loads((delivery/'VERIFY_TEST.json').read_text(encoding='utf-8-sig'))['status']=='PASS'
  result=delivery/'eval/metrics.json'
  m=json.loads(result.read_text(encoding='utf-8-sig'))
  for label,method in [('Original','original'),('Presence','presence'),('Rank Fusion','rank_fusion'),('FSR','fsr'),('SCAR','scar_s0')]:
   x[label+'_top1']=100*m['results_loc'][method]['Top1@5m']
   x[label+'_cell']=100*m['results_cell'][method]['R@1']
  x['source']=str(result)
  x['notes']='修正版追加探索性TEST；正式s0；核验PASS。'+('116查询/29场景簇，三个控制排序重合。' if domain=='npm3d' else '542查询/68房间簇，采用实际发布分母。')
 if k=='MP_KITTI':
  p=R/'FORMAL_SCAR_MP_KITTI_S0EP5_TEST_APPEND_20261104/results'
  for metric,filename,column in [('cell','table2_cell_retrieval.csv','R@1'),('top1','table3_localization.csv','top1@5m')]:
   result_path=p/filename
   results={row['method']:row[column] for row in read(result_path)}
   for name,key in [('Original','text2loc'),('Presence','presence'),('Rank Fusion','c_borda'),('FSR','sat_fine'),('SCAR','mp_formal_s0_ep5')]:
    x[name+'_'+metric]=results[key]
   sources.append(result_path)
  x['source']=str(p)
  x['notes']='新轨迹 s0 ep5；已见 TEST 上获授权的追加探索性评价，独立核验通过。历史A来自旧评价。'
 if k=='MP_PANDA':
  x['SCAR_top1']='22.15';x['SCAR_cell']='29.50';x['notes']='新轨迹s0 ep2追加探索性评价；存在一次性协议解释争议；历史A是旧轨迹。';x['source']=str(R/'SCAR_FORMAL_INCREMENTAL_20261104/mp_pandaset/results/metrics.json')
 if k=='T2L_NUSC':
  pA=R/'text2loc_five_datasets_20261026/_src_cache/nuscenes/metrics_test.json'
  pB=R/'improvement_track_user_20261027/U4/RESULTS.json'
  sources+=[pA,pB]
  x['Historical A_top1']='54.74807510468729';x['Historical A_cell']='62.17749560988788'
  x['SCAR_top1']='54.68053491827637';x['SCAR_cell']='62.191003647170064'
  x['source']=str(pA)+'; '+str(pB)
  x['notes']='已闭合：历史A=s0 ep27（7b48d6cf…），正式SCAR=B_s0 ep23（ce6fb327…）；共用同一TEST输入，5m 54.7481 vs 54.6805 与 Cell 62.1775 vs 62.1910 仅源于选头不同。'
 if k=='T2LPP_NUSC':
  p=R/'d_machine/text2locpp_nuscenes/ht/runs/eval_test/metrics.json';d=json.loads(p.read_text(encoding='utf-8'));sources.append(p)
  x['SCAR_top1']=x['Historical A_top1']=100*d['results_loc']['ht23_t2lpp_s0']['top1@5m'];x['SCAR_cell']=x['Historical A_cell']=100*d['results_cell']['ht23_t2lpp_s0']['R@1'];x['notes']='最新选轮复核A=B ep20，复用历史同头TEST，无新增评价。';x['source']=str(p)
 x['formal_status']='PENDING' if x['SCAR_top1']=='PENDING' else ('AVAILABLE_WITH_PROTOCOL_CAVEAT' if k=='MP_PANDA' else 'AVAILABLE')
 rows.append(x)
# Keep the five methods in a stable order.
order={'Text2Loc':0,'Text2Loc++':1,'MNCL':2,'MambaPlace':3,'Des4Pos corrected':4};rows.sort(key=lambda x:order[x['method']])
fields=list(rows[0]);
with (O/'RESULTS_SOURCE.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
names=['Original','Presence','Rank Fusion','FSR','Historical A','SCAR']
def fmt(v):return 'PENDING' if v=='PENDING' else f'{float(v):.2f}'
text=['# SCAR 修正版结果表（2026-11-04）','', 'TEST，单位%。正式SCAR采用定位优先VAL选轮；历史A仅为选轮参照。未运行模型、未打开TEST缓存、未重算区间，仅整理已存结果。','', '注意：这是来源对账工作表，不代表25配置均经全量独立语义验收。未完成或存在版本冲突标PENDING；不使用VAL填补TEST。','']
for met,title in [('top1','Top-1@5m'),('cell','Cell R@1')]:
 text+=['## '+title,'','| 主干 | 数据集 | 原序 | Presence | Rank Fusion | FSR | 历史A参照 | SCAR |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:text.append('| '+' | '.join([r['method'],r['dataset']]+[fmt(r[n+'_'+met]) for n in names])+' |')
 text+=['']
text+=['## 版本及缺项说明','', '- Des4Pos只列corrected：PandaSet与nuScenesFull的TEST待回执，整行PENDING；NPM3D与S3DIS采用已核验的C队列结果。KITTI采用test13；不拼接旧主干A/B。','- MP×KITTI采用新轨迹s0 ep5追加探索性结果：Top-1@5m 57.07%，Cell R@1 40.83%。','- MP×PandaSet采用新轨迹ep2：22.15%/29.50%；协议解释争议保留，不能作为无争议确认性证据。','- T2L++×nuScenes正式A=B已由新选轮记录确认，引用原同头TEST。','- Text2Loc×nuScenes两汇总读数已闭合：54.7481/62.1775来自历史A（s0 ep27，sha 7b48d6cf…），54.6805/62.1910来自正式SCAR（LSEL-v1 B_s0 ep23，sha ce6fb327…）。两者共用同一冻结TEST输入（features ae2032e9…、candidates c02accc1…、coarse 232d04b0…），5m差5条命中、Cell差1条，仅源于选头不同。本表两侧分别采用。','- 其他历史公开主干数值来自已有汇总，未因此新增验证结论；Presence等控制的具体定义随配置记录，不假设实现完全相同。','- 不在本表呈现区间或显著性，避免沿用曾错位的DELTA区间；1m/3m/10m尚未统一，不从5m反推。','- 历史A=PENDING在修正版Des4Pos表示未开展该对照，不意味着必须新增实验。','']
(O/'修正版结果表.md').write_text('\n'.join(text),encoding='utf-8')
for met,title in [('top1','TOP1_5M'),('cell','CELL_R1')]:
 with (O/(title+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['method','dataset','split']+names+['status','notes']);
  for r in rows:w.writerow([r['method'],r['dataset'],'TEST']+[r[n+'_'+met] for n in names]+[r['formal_status'],r['notes']])
assert len(rows)==25 and len({r['id'] for r in rows})==25
assert sum(r['SCAR_top1']=='PENDING' for r in rows)==2
for r in rows:
 for n in names:
  for m in ['top1','cell']:
   v=r[n+'_'+m];assert v=='PENDING' or 0<=float(v)<=100
(O/'SOURCE_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},ensure_ascii=False,indent=2),encoding='utf-8')
print('25 rows; 23 formal values (1 protocol caveat); 2 PENDING. Files:');print('\n'.join(p.name for p in O.iterdir()))
