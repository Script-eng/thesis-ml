-- Create ml_predictions table for storing model predictions
-- This table stores end-of-day closing price predictions from LSTM, RNN, and Prophet models

CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_date DATE NOT NULL,  -- The date we're predicting the close for

    -- Current market data at prediction time
    current_price DECIMAL(10, 2),

    -- Model predictions
    lstm_prediction DECIMAL(10, 2),
    lstm_confidence DECIMAL(5, 4),  -- R² score or confidence metric

    rnn_prediction DECIMAL(10, 2),
    rnn_confidence DECIMAL(5, 4),

    prophet_prediction DECIMAL(10, 2),
    prophet_confidence DECIMAL(5, 4),

    -- Ensemble prediction (weighted average or best model)
    ensemble_prediction DECIMAL(10, 2),
    ensemble_confidence DECIMAL(5, 4),

    -- Trading signal
    signal VARCHAR(10) CHECK (signal IN ('BUY', 'SELL', 'HOLD')),

    -- Metadata
    model_version VARCHAR(20),
    notes TEXT,

    -- Indexes for fast lookups
    CONSTRAINT unique_symbol_date UNIQUE (symbol, prediction_date)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ml_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON ml_predictions(prediction_time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON ml_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON ml_predictions(symbol, prediction_date);

-- Comments for documentation
COMMENT ON TABLE ml_predictions IS 'ML model predictions for end-of-day closing prices';
COMMENT ON COLUMN ml_predictions.prediction_date IS 'The trading day we are predicting the close for';
COMMENT ON COLUMN ml_predictions.prediction_time IS 'When the prediction was generated';
COMMENT ON COLUMN ml_predictions.ensemble_prediction IS 'Weighted average or best model prediction';
COMMENT ON COLUMN ml_predictions.signal IS 'Trading signal: BUY (predicted > current), SELL (predicted < current), HOLD (neutral)';
