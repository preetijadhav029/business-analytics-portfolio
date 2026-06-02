 CREATE TABLE upi_dataset (
 user_id VARCHAR(10),
    mandate_attempt_datetime DATETIME,
    day_of_week VARCHAR(10),
    hour_of_day INT,
    setup_status VARCHAR(10),
    failure_reason VARCHAR(50)
    );
    
    SELECT * FROM upi_dataset;
    
    
    
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/upimandate_dataset.csv'
INTO TABLE upi_dataset
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SELECT COUNT(*) FROM upi_dataset;

# Overall Success vs Failure Rate
SELECT 
setup_status,
COUNT(*) AS total_attempts,
ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM upi_dataset),2) AS percentage
FROM upi_dataset
GROUP BY setup_status;

# Success Rate by Hour (Core Analysis)
SELECT 
hour_of_day,
COUNT(*) AS attempts,
SUM(setup_status='Success') AS successes,
ROUND(SUM(setup_status='Success')/COUNT(*)*100,2) AS success_rate
FROM upi_dataset
GROUP BY hour_of_day
ORDER BY success_rate DESC;

# Success Rate by Day of Week
SELECT 
day_of_week,
COUNT(*) AS attempts,
SUM(setup_status='Success') AS successes,
ROUND(SUM(setup_status='Success')/COUNT(*)*100,2) AS success_rate
FROM upi_dataset
GROUP BY day_of_week;

# Identify the Best Time Window
SELECT 
day_of_week,
hour_of_day,
COUNT(*) attempts,
SUM(setup_status='Success') successes,
ROUND(SUM(setup_status='Success')/COUNT(*)*100,2) success_rate
FROM upi_dataset
GROUP BY day_of_week, hour_of_day
HAVING attempts > 10
ORDER BY success_rate DESC
LIMIT 10;


# Failure Reason Analysis 
SELECT 
failure_reason,
COUNT(*) AS failures,
ROUND(COUNT(*)*100.0/
(SELECT COUNT(*) FROM upi_dataset WHERE setup_status='Failed'),2) AS percentage
FROM upi_dataset
WHERE setup_status='Failed'
GROUP BY failure_reason
ORDER BY failures DESC;

# Chart 1 — Success Rate by Hour
SELECT 
hour_of_day,
ROUND(SUM(setup_status='Success')/COUNT(*)*100,2) AS success_rate
FROM upi_dataset
GROUP BY hour_of_day
ORDER BY hour_of_day;

# Chart 2 — Success Rate by Day
SELECT 
day_of_week,
ROUND(SUM(setup_status='Success')/COUNT(*)*100,2) AS success_rate
FROM upi_dataset
GROUP BY day_of_week;

# Chart 3 — Failure Reasons
SELECT 
failure_reason,
COUNT(*) AS failures
FROM upi_dataset
WHERE setup_status='Failed'
GROUP BY failure_reason;