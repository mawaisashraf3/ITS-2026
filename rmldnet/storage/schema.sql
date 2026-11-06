PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS datasets (
 id INTEGER PRIMARY KEY, dataset_key TEXT NOT NULL UNIQUE,
 name TEXT NOT NULL CHECK(length(name)>0), metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
 created_at TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS samples (
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
 external_id TEXT NOT NULL, image_uri TEXT NOT NULL, mask_uri TEXT NOT NULL,
 sha256 TEXT NOT NULL CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
 group_id TEXT NOT NULL, instance_supervision INTEGER NOT NULL CHECK(instance_supervision IN (0,1)),
 UNIQUE(dataset_id,external_id), UNIQUE(dataset_id,sha256), UNIQUE(id,dataset_id)
);
CREATE TABLE IF NOT EXISTS splits (
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
 name TEXT NOT NULL CHECK(name IN ('train','validation','test')),
 UNIQUE(dataset_id,name), UNIQUE(id,dataset_id)
);
CREATE TABLE IF NOT EXISTS split_memberships (
 dataset_id INTEGER NOT NULL, sample_id INTEGER PRIMARY KEY, split_id INTEGER NOT NULL,
 FOREIGN KEY(sample_id,dataset_id) REFERENCES samples(id,dataset_id) ON DELETE CASCADE,
 FOREIGN KEY(split_id,dataset_id) REFERENCES splits(id,dataset_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS experiments (
 id INTEGER PRIMARY KEY, dataset_id INTEGER NOT NULL REFERENCES datasets(id),
 run_key TEXT NOT NULL UNIQUE CHECK(length(run_key)=64), model TEXT NOT NULL,
 seed INTEGER NOT NULL CHECK(seed>=0), state_sha256 TEXT NOT NULL CHECK(length(state_sha256)=64),
 config_json TEXT NOT NULL CHECK(json_valid(config_json)), result_sha256 TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS conditions (
 id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
 beta REAL NOT NULL CHECK(beta>=0), threshold REAL NOT NULL CHECK(threshold>=0 AND threshold<=1),
 sample_count INTEGER NOT NULL CHECK(sample_count>0), elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds>=0),
 tn INTEGER NOT NULL CHECK(typeof(tn)='integer' AND tn>=0), fp INTEGER NOT NULL CHECK(typeof(fp)='integer' AND fp>=0),
 fn INTEGER NOT NULL CHECK(typeof(fn)='integer' AND fn>=0), tp INTEGER NOT NULL CHECK(typeof(tp)='integer' AND tp>=0), UNIQUE(experiment_id,beta)
);
CREATE TABLE IF NOT EXISTS observations (
 condition_id INTEGER NOT NULL REFERENCES conditions(id) ON DELETE CASCADE,
 sample_id INTEGER NOT NULL REFERENCES samples(id),
 tn INTEGER NOT NULL CHECK(typeof(tn)='integer' AND tn>=0), fp INTEGER NOT NULL CHECK(typeof(fp)='integer' AND fp>=0),
 fn INTEGER NOT NULL CHECK(typeof(fn)='integer' AND fn>=0), tp INTEGER NOT NULL CHECK(typeof(tp)='integer' AND tp>=0),
 PRIMARY KEY(condition_id,sample_id)
);
CREATE TRIGGER IF NOT EXISTS observation_split_guard BEFORE INSERT ON observations
WHEN NOT EXISTS (
 SELECT 1 FROM conditions c JOIN experiments e ON e.id=c.experiment_id
 JOIN samples s ON s.dataset_id=e.dataset_id
 JOIN split_memberships m ON m.sample_id=s.id
 JOIN splits p ON p.id=m.split_id
 WHERE c.id=NEW.condition_id AND s.id=NEW.sample_id AND p.name='test'
)
BEGIN SELECT RAISE(ABORT,'observation must reference a test sample in the experiment dataset'); END;
CREATE TRIGGER IF NOT EXISTS observation_update_guard BEFORE UPDATE OF condition_id,sample_id ON observations
WHEN NOT EXISTS (
 SELECT 1 FROM conditions c JOIN experiments e ON e.id=c.experiment_id
 JOIN samples s ON s.dataset_id=e.dataset_id
 JOIN split_memberships m ON m.sample_id=s.id
 JOIN splits p ON p.id=m.split_id
 WHERE c.id=NEW.condition_id AND s.id=NEW.sample_id AND p.name='test'
)
BEGIN SELECT RAISE(ABORT,'observation must reference a test sample in the experiment dataset'); END;
CREATE TABLE IF NOT EXISTS artifacts (
 id INTEGER PRIMARY KEY, experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
 relative_path TEXT NOT NULL, kind TEXT NOT NULL,
 sha256 TEXT NOT NULL CHECK(length(sha256)=64), bytes INTEGER NOT NULL CHECK(bytes>=0),
 UNIQUE(experiment_id,relative_path)
);
CREATE INDEX IF NOT EXISTS idx_conditions_experiment ON conditions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_samples_group ON samples(dataset_id,group_id);
CREATE INDEX IF NOT EXISTS idx_observations_sample ON observations(sample_id);
CREATE VIEW IF NOT EXISTS metrics AS
SELECT e.run_key,e.model,e.seed,c.id AS condition_id,c.beta,c.threshold,c.sample_count,
 c.tn,c.fp,c.fn,c.tp,c.elapsed_seconds,
 1.0*c.tp/NULLIF(c.tp+c.fp,0) AS precision,
 1.0*c.tp/NULLIF(c.tp+c.fn,0) AS recall,
 2.0*c.tp/NULLIF(2*c.tp+c.fp+c.fn,0) AS f1,
 1.0*(c.tp+c.tn)/NULLIF(c.tp+c.tn+c.fp+c.fn,0) AS accuracy,
 1.0*c.tp/NULLIF(c.tp+c.fp+c.fn,0) AS lane_iou,
 (COALESCE(1.0*c.tp/NULLIF(c.tp+c.fp+c.fn,0),0)+
  COALESCE(1.0*c.tn/NULLIF(c.tn+c.fp+c.fn,0),0)) /
 NULLIF((CASE WHEN c.tp+c.fp+c.fn>0 THEN 1 ELSE 0 END)+
        (CASE WHEN c.tn+c.fp+c.fn>0 THEN 1 ELSE 0 END),0) AS miou
FROM conditions c JOIN experiments e ON e.id=c.experiment_id;
PRAGMA user_version = 1;
