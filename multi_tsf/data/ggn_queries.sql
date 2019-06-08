select max(CONTEXT_CONFIG_ID) from CONTEXT_CONFIG cc 
where EFFECTIVE_TS < '2019-03-18 12:00:00' and CONFIG_STATUS_NM='PRODUCTION';


select * from CONTEXT_CONFIG cc where CONTEXT_CONFIG_ID=103648;

select
d.* from CONTEXT_CONFIG cc
inner join DEMAND d on d.DEMAND_CONTEXT_ID=cc.CONTEXT_DEMAND_ID where cc.CONTEXT_CONFIG_ID=103648
and DEMAND_TYPE_C='CURRENT';

select
d.* from DEMAND d
where WORK_SET_ID=125723 and demand_type_c='CURRENT' limit 100;


select * from context_config limit 100;

SELECT context_demand_id FROM (
 SELECT context_demand_id, 
  ROW_NUMBER() OVER (PARTITION BY context_demand_id ORDER BY context_config_id DESC) rn
 FROM context_config
) tmp WHERE rn = 1 limit 100;


-- distinct demand types
select distinct(demand_type_c) from (select * from demand limit 1000000) tmp;

-- work_set to attributes
select * from work_set_attribute_map a join work_set_attribute b 
on a.work_set_attribute_id = b.work_set_attribute_id;

-- skill_nm to work_set_id
select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id
from work_skill_log a join work_set_log b 
on a.work_skill_id=b.work_skill_id;

-- counts by skill_nm
select skill_display_nm, count(skill_display_nm) as work_set_count
from work_skill_log a join work_set_log b 
on a.work_skill_id=b.work_skill_id group by skill_display_nm order by work_set_count DESC;

--get half-hourly current workload for given work_set_id
select start_ts,
       work_set_id,
       demand_context_id,
       count(*) as row_count,
       sum(item_count_nb) as item_count_nb_sum,
       sum(amount_nb) as amount_nb_sum
from demand 
where demand_type_c = 'CURRENT' 
      and work_set_id = 125723
      and demand_context_id in (select distinct on (start_ts, work_set_id) demand_context_id
                                  from demand where demand_type_c = 'CURRENT' and work_set_id = 125723)
group by work_set_id, start_ts, demand_context_id; 

-------------------------DEMAND--------------------------
--half-hourly workload by work_set_id and skill_nm
SELECT ts.time_stamp, js.*, skill_nm.skill_display_nm
FROM
    (SELECT *
    FROM   generate_series(timestamp '2018-12-31'::timestamp
                         , timestamp '2019-01-31'::timestamp
                         , interval  '30 min'::interval) time_stamp
    WHERE extract(hour from time_stamp) >= 6
    and extract(hour from time_stamp) <= 23) ts

LEFT JOIN
    (select start_ts, work_set_id, avg(item_count_nb_sum) as item_count_nb_sum, avg(estimated_completion_time_hrs) as estimated_completion_time_hrs
    from (
          select 
           start_ts,
           work_set_id,
           demand_context_id,
           sum(item_count_nb) as item_count_nb_sum,
           sum(amount_nb)/(60*60*1000) as estimated_completion_time_hrs
            from demand 
            where demand_type_c = 'CURRENT' 
            and start_ts > '2018-12-31'
            and start_ts <= '2019-01-31' 
          group by work_set_id, start_ts, demand_context_id) A
    group by A.work_set_id, A.start_ts) js
          
          on ts.time_stamp = js.start_ts

LEFT JOIN
  (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id
      from work_skill_log a JOIN work_set_log b 
      on a.work_skill_id=b.work_skill_id) skill_nm 

ON js.work_set_id = skill_nm.work_set_id;


--Summed over all skill_nm
SELECT start_ts, skill_display_nm, sum(item_count_nb_sum) as total_item_count, sum(estimated_completion_time_hrs) as total_est_completion_hrs
FROM
    ((SELECT *
    FROM   generate_series(timestamp '2018-12-31'::timestamp
                         , timestamp '2019-01-31'::timestamp
                         , interval  '30 min'::interval) time_stamp
    WHERE extract(hour from time_stamp) >= 6
    and extract(hour from time_stamp) <= 23) ts

LEFT JOIN
    (select start_ts, work_set_id, avg(item_count_nb_sum) as item_count_nb_sum, avg(estimated_completion_time_hrs) as estimated_completion_time_hrs
    from (
          select 
           start_ts,
           work_set_id,
           demand_context_id,
           sum(item_count_nb) as item_count_nb_sum,
           sum(amount_nb)/(60*60*1000) as estimated_completion_time_hrs
            from demand 
            where demand_type_c = 'CURRENT' 
            and start_ts > '2018-12-31'
            and start_ts <= '2019-01-31' 
          group by work_set_id, start_ts, demand_context_id) A
    group by A.work_set_id, A.start_ts) js
          
          on ts.time_stamp = js.start_ts

LEFT JOIN
  (select distinct on (skill_display_nm, work_set_id) skill_display_nm, work_set_id
      from work_skill_log a JOIN work_set_log b 
      on a.work_skill_id=b.work_skill_id) skill_nm 

ON js.work_set_id = skill_nm.work_set_id) ws_time_series group by start_ts, skill_display_nm;


-----------------Work Item Log-----------------
SELECT wil_cs.received_ts_rounded, wil_cs.work_set_id, wil_cs.rec_item_count
FROM (select received_ts_rounded, work_set_id, count(work_item_id) as rec_item_count 
FROM work_item_log_cs 
WHERE received_ts_rounded >= '2019-01-01'
AND received_ts_rounded <= '2019-03-23'
GROUP BY received_ts_rounded, work_set_id) as wil_cs;



------Getting Representative day forecast---------
select curr.past_sample_date
        , ref.received_ts_rounded as past_received_ts_rounded
        , ref.work_set_id
        , curr.start_date
        , curr.start_date + cast(ref.received_ts_rounded as time)  as current_received_ts_rounded
        , ref.rec_item_count
from
(SELECT DATE(start_ts) as start_date,DATE(sample_ts) as past_sample_date FROM (
           SELECT context_config_id, modified_ts, effective_ts, start_ts, end_ts, sample_ts,
            ROW_NUMBER() OVER (PARTITION BY DATE(start_ts) ORDER BY  effective_ts DESC) rn
           FROM context_config where config_status_nm = 'PRODUCTION'
           and start_ts >= '2017-01-01' 
            ) tmp WHERE rn = 1) curr
join 
(select work_set_id,
       received_ts_rounded,
       count(work_item_id) as rec_item_count
from work_item_log_cs where received_ts_rounded >= '2017-01-01'
group by work_set_id, received_ts_rounded) ref 
on curr.past_sample_date = DATE(ref.received_ts_rounded); 

select * from interval_grouped_work_items limit 10;
