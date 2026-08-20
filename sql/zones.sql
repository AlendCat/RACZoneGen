SELECT
    z.ZONE_OID,
    z.NAME AS [Name],
    z.BOUNDARY_POLYGON.STEnvelope().STPointN(1).STX AS [MinX],
    z.BOUNDARY_POLYGON.STEnvelope().STPointN(1).STY AS [MinY],
    z.BOUNDARY_POLYGON.STEnvelope().STPointN(3).STX AS [MaxX],
    z.BOUNDARY_POLYGON.STEnvelope().STPointN(3).STY AS [MaxY],
    z.SPEED_LIMIT AS [SpeedLimit]
FROM [msmodel].[dbo].[ZONE] z WITH (NOLOCK)
WHERE z.IS_ACTIVE = 1;