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
select skill_display_nm, work_set_id 
from work_skill_log a join work_set_log b 
on a.work_skill_id=b.work_skill_id;


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
