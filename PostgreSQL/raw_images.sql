CREATE TABLE raw_images (
    id           BIGSERIAL PRIMARY KEY,
    uploaded_at  TIMESTAMPTZ DEFAULT now(),
    filename     TEXT,
    content_type TEXT,
    width        INTEGER,
    height       INTEGER,
    bytes        BYTEA NOT NULL
);
