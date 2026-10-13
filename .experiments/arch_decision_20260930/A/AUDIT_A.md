结论：无封存测试集（zgz兼选点与主表）。local每段GT复位、global单run；指标皆根对齐。§9不符：噪声「逐维独立」不适用于定向档。混杂（热启短训/单种子/batch/骨干）。3407/3408同配置换seed，S37无热启。
证据：splits_semkine.json；dataset.py:475-476,243-250；eval_track.py:149-159,221-225；make_s36_row.py:54-89；model.py:923,1460,1597；training_metadata.json。
否决：跨臂不可单变量归因。
最小实验：S37选中点非选点集重评RA。
