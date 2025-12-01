CREATE TABLE visdrone_files (
    id         SERIAL PRIMARY KEY,
    split      TEXT        NOT NULL,  -- 'train' 或 'val'
    file_type  TEXT        NOT NULL,  -- 'image' 或 'annotation'
    filename   TEXT        NOT NULL,  -- 檔名
    rel_path   TEXT        NOT NULL,  -- 相對路徑（從 datasets/ 底下開始）
    abs_path   TEXT        NOT NULL,  -- 絕對路徑（完整 Windows 路徑）
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 避免重複插入同一個檔案
CREATE UNIQUE INDEX uq_visdrone_file
ON visdrone_files(split, file_type, rel_path);
