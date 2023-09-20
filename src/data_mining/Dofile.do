clear
use "panel_cpr_all"


*Part 1: Data cleaning*

*Define labels
label define organs 0 "POEs" 1 "SOEs"
label value organ_type organs
label define crdt 0 "black" 1 "green" 2 "red"
label value credit crdt


 
*encodes
encode province_con, g(province)
encode IndustryCode, g(ind)

*********** Global Variables *************
global c "black red"
global lc "l.black l.red"
global llc "l.l.black l.l.red"
global info "log_age log_employee"
global info_n "log_employee log_age"
global d "black_daughters red_daughters"
global d "log_black_Subsidiary log_red_Subsidiary"


*claim the panel
xtset ID year


*Part 2: Descriptive statistics

tab credit

*Firm performance indicators
global log_x_firm "log_patents_apply log_TotalAssets log_TotalLiability log_IntangibleAsset log_NetProfit log_OperatingEvenue log_OperatingCost log_OperationProfit log_asset_turnover log_employee log_RD log_age"
pwcorr $log_x_firm 
tabstat $log_x_firm, stats(N mean sd min p50 max)



****************************************************************************
********************Logit: performance -> credit*************************
****************************************************************************

/// (1) Basic model
logit red log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type log_employee log_age i.year i.ind i.province,  vce(robust) 

*** RONUSTNESS TEST ***

/// (2) Add ever red
logit red ever_red log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type  log_employee log_age i.year i.ind i.province, vce(robust)  

/// (3) Black dropped
logit red log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type  log_employee log_age i.year i.ind i.province if black!=1, vce(robust)

/// (4) Add province-year FEs
logit red ever_red log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type  log_employee log_age i.ind i.year##i.province, vce(robust)  


*** RONUSTNESS TEST: Only green control group ***





  
****************************************************************************
****************************OLS: Credit -> Performance**********************
****************************************************************************
est clear

**************************** OLS with credit ***************************
local x_3 "log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy"

/// X = log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type log_employee log_age i.year i.ind i.province

foreach y of local x_3 {
    local x_list : list x_3 - y 
    reg `y' $c $lc `x_list' $info_n organ_type i.year i.province i.ind, cluster(ID) 
	est store pn`y'
}

**************************** OLS with credit with LAG ***************************

local x_3 "log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy"

// Create an empty list to store the lagged variable names
local lagged_vars

// Generate lagged variables and add their names to the list
foreach var of local x_3 {
    gen lag_`var' = L.`var' if !missing(`var')
    
    // Append the lagged variable name to the list
    local lagged_vars "`lagged_vars' lag_`var'"
}

// Display the list of lagged variable names
di "`lagged_vars'"

foreach y of local x_3 {
    local lag_var lag_`y' // Define the lagged variable to exclude
    
    // Create x_list without the excluded variable
    local x_list ""
    foreach var of local lagged_vars {
        if "`var'" != "`lag_var'" {
            local x_list "`x_list' `var'"
        }
    }
    
    // Run the regression
    reg `y' $c $lc `x_list' $info_n organ_type i.year i.province i.ind, cluster(ID) 
    
    // Store the results
    est store pn`y'
}






*esttab pn* using "re.log.rtf", p replace cells(b(star fmt(3)) se(par fmt(2)))   ///
*   legend label varlabels(_cons constant)               ///
*   stats(r2 df_r bic, fmt(3 0 1) label(R-sqr dfres BIC))

*reg log_labour_productivity red black log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type i.year i.province i.ind, cluster(ID) 

*reg log_labour_productivity log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type i.year i.province i.ind, cluster(ID) 



/// Effect with different control groups

/// Red vs. black and green
reg log_real_subsidy red L.log_real_OperatingCost L.log_real_NetProfit L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if real_subsidy>=0, cluster(ID)

/// Red vs. green
reg log_real_subsidy red L.log_real_OperatingCost L.log_real_NetProfit L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if real_subsidy>=0 & black!=1, cluster(ID)

/// Red vs. black
reg log_real_subsidy red L.log_real_OperatingCost L.log_real_NetProfit L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if real_subsidy>=0 & green!=1, cluster(ID)

/// Black vs. red and green ***
reg log_real_subsidy black L.log_real_OperatingCost L.log_real_NetProfit L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if real_subsidy>=0, cluster(ID)


reg log_real_subsidy red L.log_real_OperatingCost L.log_real_NetProfit L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if real_subsidy>=0 & green!=1, cluster(ID)


reg log_real_subsidy red black log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply organ_type i.year i.province i.ind if real_subsidy>=0, cluster(ID)

*red black ever_red i.year i.province i.ind, cluster(ID)
*sum log_real_subsidy real_subsidy, d


/// **New added: Net Profit

/// Red vs. black and green
reg log_real_NetProfit red L.log_real_OperatingCost L.log_real_subsidy L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind, cluster(ID)

/// Red vs. green
reg log_real_NetProfit red L.log_real_OperatingCost L.log_real_subsidy L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if black!=1, cluster(ID)

/// Red vs. black
reg log_real_NetProfit red L.log_real_OperatingCost L.log_real_subsidy L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind if green!=1, cluster(ID)

/// Black vs. red and green ***
reg log_real_NetProfit black L.log_real_OperatingCost L.log_real_subsidy L.log_real_RD log_patents_apply L.organ_type L.log_employee L.log_age  i.year i.province i.ind, cluster(ID)









****************************************************************************
****************************OLS: with daughters********************************
****************************************************************************

/// ### Add  daughters
logit red total_daughters red_daughters black_daughters log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_real_subsidy organ_type  log_employee log_age i.year i.ind i.province, vce(robust)




*local x_3 "log_real_OperatingCost log_real_NetProfit log_real_RD log_patents_apply log_asset_turnover log_op_profit_margin"

*foreach y of local x_3 {
*    local x_list : list x_3 - y 
*    reg `y' $c $d `x_list' $info_n organ_type i.year i.province i.ind, cluster(ID)
*	est store pn`y'
*}







