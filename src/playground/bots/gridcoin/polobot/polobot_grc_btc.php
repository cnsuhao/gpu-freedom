<?php
	// A simple arbitrage trading bot (a taker bot)
	// (c) by 2017 dangermouse, GPL licence
	// API reference is at https://poloniex.com/support/api/

	
	require_once("poloniex_api.php");
        require_once("conf/config.inc.php");	

	$date = date('Y-m-d H:i:s');
	echo "$date\n";
	

	// fee structure
	$fee_maker = 0.0015;
	$fee_taker = 0.0025;
	
	// currency to be traded
	$currency_1 = "GRC";
	$max_tradable_1 = 100; // maximum amount tradable in currency 1

	$currency_2 = "BTC";
        $max_tradable_2 = 0.01; // maximum amount tradable in currency 2
		
	$currency_ref = "USDT"; // tether as reference currency to maximize portfolio
        

	// currency pairs
	$curpair_1_2 = $currency_2 . "_" . $currency_1;
	$curpair_2_ref = $currency_ref . "_" . $currency_2;
        
        //echo "API key: ";
        //echo $my_api_key;
        //echo "\n";
        //echo "API secret: ";
        //echo $my_api_secret;
        //echo "\n";

        
	$api = new poloniex($my_api_key, $my_api_secret);
	
	// 0. cancel existing orders lying around from previous bot calls 
	// (they lay around for example if the order could be only partially fullfilled)
	// get_open_orders($pair), retrieves ordernumber
	// iterate over and do cancel_order($pair, $order_number)
	
        echo "Retrieving open orders...\n";
        $openorders = $api -> get_open_orders($curpair_1_2);
	//print_r($openorders);
        echo count($openorders);
        echo "\n";
        /*
        for (int i=0; i<count($openorders); i++) {
                   if ($openorders[i]["amount"])<=$max_tradable_1) {                        
                        echo "Cancelling order ";
                        echo $openorders[i]["orderNumber"];
                        echo "\n";
			$api->cancel_order($curpair_1_2, $openorders[i]["orderNumber"]);
                   } else {
                        echo "Order ";
                        echo $openorders[i]["orderNumber"];
                        echo " not cancelled due to high amount\n";

		   }
	}
        */
           
	
	// 1. retrieve current prices
	// TODO: retrieve also bid and ask to be more accurate (using lowestAsk and highestBid)
	
        //$myres = $api->get_trading_pairs();
        //echo "*\n";
        //print_r($myres);
        //echo "*\n";
        $myres_curpair_1_2   = $api->get_ticker($curpair_1_2);
        $myres_curpair_2_ref = $api->get_ticker($curpair_2_ref); 

        $price_1_in_2 = $myres_curpair_1_2["last"]; 
        $price_2_in_ref = $myres_curpair_2_ref["last"];
	$price_1_in_ref = $price_1_in_2 * $price_2_in_ref;
	
	echo "$price_1_in_ref $currency_1/$currency_ref      $price_2_in_ref $currency_ref/$currency_2      $price_1_in_2 $currency_2/$currency_1   \n";
	
	
	// 2. retrieve our current balance in currency 1 and 2
	//    and calculate current portfolio value in reference currency
	sleep(1);
        $balances = $api->get_balances();
	//print_r($balances);
        //echo $balances["GRC"]["available"];
        //echo "\n";
        //echo $balances["BTC"]["available"];
        //echo "\n";

        $balance_cur_1 = min($balances[$currency_1]["available"],$max_tradable_1);
	$balance_cur_2 = min($balances[$currency_2]["available"],$max_tradable_2);
	
        

	$cur_portfolio_value_ref = ($balance_cur_1 * $price_1_in_ref) + ($balance_cur_2 * $price_2_in_ref);
	
	echo "$balance_cur_1 $currency_1  +   $balance_cur_2 $currency_2      ->    $cur_portfolio_value_ref $currency_ref\n";
	
        
	// 3. now go through order book and see which order would maximize our portfolio value in ref currency
	$orderbook = $api->get_order_book($curpair_1_2);
	//print_r($orderbook);
        if ($orderbook["isFrozen"]==1) die("Frozen orderbook!");
        
        $bestbid = $orderbook["bids"][0][0]; // best offer when we want to sell
	$bestask = $orderbook["asks"][0][0]; // best offer when we want to buy
        
        echo "bestbid: $bestbid     bestask: $bestask $currency_2/$currency_1\n";	

        $tradable_amount_bid = min($max_tradable_1, $orderbook["bids"][0][1]);
        $tradable_amount_ask = min($max_tradable_1, $orderbook["asks"][0][1]);
        echo "tradable amount bid: $tradable_amount_bid  ask: $tradable_amount_ask\n";
        
	// 4. now we check if selling the tradable amount makes our portfolio look better in refcurrency
	if (($balance_cur_1 - $tradable_amount_bid)>0) {
		$new_portfolio_value_ref_sell = (($balance_cur_1 - $tradable_amount_bid) * $price_1_in_ref) + // balance in currency 1 is decreased
		                           (($balance_cur_2 + $tradable_amount_bid*$bestbid  ) * $price_2_in_ref) // balance in currency 2 is increased
		                           - ($tradable_amount_bid*$bestbid*$fee_taker) * $price_2_in_ref;  // fees
								   
	} else {
		$new_portfolio_value_ref_sell = 0;
        }
	
	// 5. specularly we check if buying the tradable amount makes our portfolio look better in refcurrency
	if (($balance_cur_2 - $tradable_amount_sell*$bestask)>0) {
		$new_portfolio_value_ref_buy = (($balance_cur_1 + $tradable_amount_sell) * $price_1_in_ref) + // balance in currency 1 is increased
		                           (($balance_cur_2 - $tradable_amount_sell*$bestask  ) * $price_2_in_ref) // balance in currency 2 is decreased
		                           - ($tradable_amount_sell*$bestask*$fee_taker) * $price_2_in_ref;  // fees
								   
	} else {
		$new_porfolio_value_ref_buy=0;
        }
        
        echo "new portfolio sell: $new_portfolio_value_ref_sell  buy: $new_portfolio_value_ref_buy $currency_ref\n";	
	// 6. now comes the decision what to do
    if (  ($new_portfolio_value_ref_sell <= $cur_portfolio_value_ref) && ($new_portfolio_value_ref_buy <= $cur_portfolio_value_ref) ) {
			// we do nothing here!!
			echo "no existing order is appealing to us...\n";
	} else {
			if ($new_portfolio_value_ref_buy>$new_portfolio_value_ref_sell) {
					// we do buy
					echo "We do buy $tradable_amount_ask $currency_1\n";
					echo "Portfolio value should go to $new_portfolio_value_ref_buy $currency_ref\n";
					//$api->buy($curpair_1_2, $bestask, $tradable_amount_ask);
			} else {
					// we do sell
					echo "We do sell $tradable_amount_sell $currency_1\n";
					echo "Portfolio value should go to $new_portfolio_value_ref_sell $currency_ref\n";
					//$api->sell($curpair_1_2, $bestbid, $tradable_amount_sell);
			}	
			
	}
	
	echo "Bot iteration over... \n\n";

?>
