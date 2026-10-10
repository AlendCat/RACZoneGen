SELECT
    CONVERT(VARCHAR(19), mshist.dbo.displayLocalTime(TIMESTAMP_UTC), 120) AS [Time],
    [helevel] AS [Level],
    [Location_X] AS [X],
    [Location_Y] AS [Y],
    TRY_CONVERT(FLOAT, PAYLOAD) AS [Payload],
    0 AS [Flag],
    0 AS [Score]
FROM [mshist].[dbo].[HEALTH_EVENT]
WHERE event_Number IN (697, 698, 699, 700, 777, 779)
  AND CLASS_ID = 'Activate'
  AND mshist.dbo.displayLocalTime(TIMESTAMP_UTC)
      > DATEADD(MINUTE, -{time_cutoff_minutes}, GETDATE());
