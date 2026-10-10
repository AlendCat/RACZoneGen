SELECT
    ln.LANE_OID,
    p.ppt.STX AS X,
    p.ppt.STY AS Y,
    'Right' AS Segment,
    p.POINT_NR
FROM [msmodel].[dbo].[LANE] ln WITH (NOLOCK)
CROSS APPLY (
    SELECT
        ln.RIGHT_EDGE_POINTS.STPointN(n) AS ppt,
        n - 1 AS POINT_NR
    FROM (
        SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
        FROM master..spt_values WITH (NOLOCK)
    ) AS numbs
    WHERE numbs.n <= ln.RIGHT_EDGE_POINTS.STNumPoints()
) AS p
WHERE ln.IS_ACTIVE = 1
  AND ln.AUTONOMOUS = 1

UNION ALL

SELECT
    ln.LANE_OID,
    p.ppt.STX AS X,
    p.ppt.STY AS Y,
    'Left' AS Segment,
    p.POINT_NR
FROM [msmodel].[dbo].[LANE] ln WITH (NOLOCK)
CROSS APPLY (
    SELECT
        ln.LEFT_EDGE_POINTS.STPointN(n) AS ppt,
        n - 1 AS POINT_NR
    FROM (
        SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
        FROM master..spt_values WITH (NOLOCK)
    ) AS numbs
    WHERE numbs.n <= ln.LEFT_EDGE_POINTS.STNumPoints()
) AS p
WHERE ln.IS_ACTIVE = 1
  AND ln.AUTONOMOUS = 1;
