SELECT
    CONVERT(VARCHAR(19), mshist.dbo.displayLocalTime(he.TIMESTAMP_UTC), 120) AS [Time],
    he.helevel AS [Level],
    he.Location_X AS [X],
    he.Location_Y AS [Y],
    m.name AS [Machine],
    TRY_CONVERT(FLOAT, he.PAYLOAD) AS [Payload],
    0 AS [Flag],
    0 AS [Score]
FROM [mshist].[dbo].[HEALTH_EVENT] he
LEFT JOIN [mshist].[dbo].[V_MACHINE] m WITH (NOLOCK)
    ON m.OID = he.MACHINE_OID
WHERE he.event_Number IN (697, 698, 699, 700, 777, 779)
  AND he.CLASS_ID = 'Activate'
  AND mshist.dbo.displayLocalTime(he.TIMESTAMP_UTC)
      > DATEADD(MINUTE, -{time_cutoff_minutes}, GETDATE());
