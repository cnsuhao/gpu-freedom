<?php
	// A simple arbitrage trading bot (a taker bot)
	// (c) by 2017 dangermouse, GPL licence
	// API reference is at https://poloniex.com/support/api/

	
	require_once("../lib/poloniex_api.php");
        require_once("../lib/bittrex_api.php");
        require_once("conf/config.inc.php");	

  $iter = 0;
  while ($iter<3) {

	$date = date('Y-m-d H:i:s');
	echo "$date  iter $iter\n";
	

	// fee structure
	$fee_polo_maker = 0.0015;
	$fee_polo_taker = 0.0025;
        
        $fee_rex_maker  = 0.0015;
        $fee_rex_taker  = 0.0025;

        $trans_treshold_in_ref = 2.3; // transactions have to be at least
                                    // this amount in $currency_ref
	
	// currency to be arbitraged
	$currency_1 = "GRC";
	$max_tradable_1 = 1000; // maximum amount tradable in currency 1

	$currency_2 = "BTC"; // currency 2
        $max_tradable_2 = 0.01; // maximum amount tradable in currency 2

        $currency_ref = "USDT"; // used to get an idea of portfolio value
        

	// currency pairs
	$curpair_1_2 = $currency_2 . "_" . $currency_1;
	$curpair_2_ref = $currency_ref . "_" . $currency_2;
        $curpair_1_2_rex = $currency_2 . "-" . $currency_1;
        $curpair_2_ref_rex = $currency_ref . "-" . $currency_2;
        /*
        echo "API poloniex key: ";
        echo $poloniex_api_key;
        echo "\n";
        echo "API poloniex secret: ";
        echo $poloniex_api_secret;
        echo "\n";
        
        echo "API bittrex key: ";
        echo $bittrex_api_key;
        echo "\n";
        echo "API bittrex secret: ";
	echo $bittrex_api_secret;
        echo "\n";
        */
        
	$api_polo = new poloniex($poloniex_api_key, $poloniex_api_secret);
        $api_rex  = new bittrex_api($bittrex_api_key, $bittrex_api_secret);	

        
	// 0. cancel existing orders lying around from previous bot calls 
	// (they lay around for example if the order could be only partially fullfilled)
	// get_open_orders($pair), retrieves ordernumber
	// iterate over and do cancel_order($pair, $order_number)
        echo "Retrieving open orders on Poloniex...\n";
        $openorders = $api_polo->get_open_orders($curpair_1_2);
	echo "* openorders poloniex result: \n";
        print_r($openorders);
        echo "*\n";
        echo "open orders ";
        echo count($openorders[0]);
        echo "\n";
        
        if (count($openorders[0])>0) { //hack due to API inconsistency
            $i=0;   
            while ($i<count($openorders)) {
                 
                   if (($openorders[$i]["amount"])<=$max_tradable_1) {                        
                        //todo: check me
                        echo "Cancelling Polo order ";
                        echo $openorders[$i]["orderNumber"];
                        echo "\n";
			$api_polo->cancel_order($curpair_1_2, $openorders[$i]["orderNumber"]);
                   } else {
                        echo "Polo order ";
                        echo $openorder[$i]["orderNumber"];
                        echo " not cancelled due to high amount, not set by this bot\n";
		   }
                 
              $i=$i+1;
	    }
        }
	//echo "Retrieving open orders on bittrex";
        $openorders_rex = $api_rex->getOpenOrders($curpair_1_2_rex);
        echo "* openorders bittrex result: \n";
        print_r($openorders_rex);
        echo "*\n"; 
        echo count($openorders_rex);
        echo "\n";
        $i=0;
        while ($i<count($openorders_rex)) {
                $orderid_rex = $openorders_rex[$i]->OrderUuid;
                $quantity_rex = $openorders_rex[$i]->Quantity;
                if ($quantity_rex<=$max_tradable_1) {
                     echo "Cancelling Rex order $orderid_rex ($quantity_rex $currency_1) \n";
                     $api_rex->cancel($orderid_rex);
                } else {
                     echo "Not cancelling Rex order $orderid_rex ($quantity_rex $currency_1 because not done by bot \n";
                }
                $i=$i+1;
        }
        
	
	// 1. retrieve current prices
	// TODO: retrieve also bid and ask to be more accurate (using lowestAsk and highestBid)
	
        //$myres = $api_polo->get_trading_pairs();
        //echo "*\n";
        //print_r($myres);
        //echo "*\n";
        $myres_curpair_1_2   = $api_polo->get_ticker($curpair_1_2);
        $myres_curpair_2_ref = $api_polo->get_ticker($curpair_2_ref); 

        $price_1_in_2 = $myres_curpair_1_2["last"]; 
        $price_2_in_ref = $myres_curpair_2_ref["last"];
	$price_1_in_ref = $price_1_in_2 * $price_2_in_ref;
	
        echo "Polo: $price_1_in_ref $currency_1/$currency_ref  $price_2_in_ref $currency_ref/$currency_2   $price_1_in_2 $currency_2/$currency_1\n";
	
        //$res = $api_rex->getMarkets();
        //print_r($res); // format for markets is BTC-GRC
        $myres_curpair_1_2_rex = $api_rex->getTicker($curpair_1_2_rex);
        //print_r($myres_curpair_1_2_rex);
        $myres_curpair_2_ref_rex = $api_rex->getTicker($curpair_2_ref_rex);
        //print_r($myres_curpair_2_ref_rex);        

        $price_1_in_2_rex = $myres_curpair_1_2_rex->Last;
        $price_2_in_ref_rex = $myres_curpair_2_ref_rex->Last;
        $price_1_in_ref_rex = $price_1_in_2_rex * $price_2_in_ref_rex;

        echo "Rex : $price_1_in_ref_rex $currency_1/$currency_ref $price_2_in_ref_rex  $currency_ref/$currency_2 $price_1_in_2_rex $currency_2/$currency_1\n";
	echo "\n";
        
	// 2. retrieve our current balance in currency 1 and 2
	//    and calculate current portfolio value in reference currency
	//sleep(1);
        $balances = $api_polo->get_balances();
	//print_r($balances);
        //echo $balances["GRC"]["available"];
        //echo "\n";
        //echo $balances["BTC"]["available"];
        //echo "\n";

        $balance_cur_1 = min($balances[$currency_1]["available"],$max_tradable_1);
	$balance_cur_2 = min($balances[$currency_2]["available"],$max_tradable_2);
	$cur_portfolio_value_ref = ($balance_cur_1 * $price_1_in_ref) + ($balance_cur_2 * $price_2_in_ref);
	echo "Polo: $balance_cur_1 $currency_1  +   $balance_cur_2 $currency_2      ->    $cur_portfolio_value_ref $currency_ref\n";
	
        $balances_rex_1 = $api_rex->getBalance($currency_1)->Available;
        $balances_rex_2 = $api_rex->getBalance($currency_2)->Available;
        //print_r($balances_rex_1);
        $balance_cur_1_rex = min($balances_rex_1, $max_tradable_1);
        $balance_cur_2_rex = min($balances_rex_2, $max_tradable_2);
        $cur_portfolio_value_ref_rex = ($balance_cur_1_rex * $price_1_in_ref_rex) + ($balance_cur_2_rex * $price_2_in_ref_rex);
        echo "Rex : $balance_cur_1_rex $currency_1   +   $balance_cur_2_rex $currency_2   -> $cur_portfolio_value_ref_rex $currency_ref\n";
        echo "\n";
        
	// 3. now go through order book of polo and rex and see which order would make a good arbitrage
	$orderbook = $api_polo->get_order_book($curpair_1_2);
	//print_r($orderbook);
        if ($orderbook["isFrozen"]==1) die("Poloniex frozen orderbook!");
        
        $bestbid = $orderbook["bids"][0][0]; // best offer when we want to sell
	$bestask = $orderbook["asks"][0][0]; // best offer when we want to buy
        $bidqty  = $orderbook["bids"][0][1];
        $askqty  = $orderbook["asks"][0][1];
        echo "Polo: bestbid: $bestbid     bestask: $bestask $currency_2/$currency_1\n";	
        echo "Polo:  bidqty: $bidqty       askqty: $askqty $currency_1\n";
        $tradable_amount_bid = min($balance_cur_1, $bidqty);
        $tradable_amount_ask = min($balance_cur_2/$bestask, $askqty);
        echo "Polo: tradable amount bid: $tradable_amount_bid  ask: $tradable_amount_ask $currency_1\n";
        echo "-------------------------\n";
        
        $orderbook_bid_rex = $api_rex->getOrderBook($curpair_1_2_rex, "buy" /* or buy or sell*/, 1 /*market depth*/);
        //print_r($orderbook_bid_rex[0]);
        $orderbook_ask_rex = $api_rex->getOrderBook($curpair_1_2_rex, "sell", 1);
        //print_r($orderbook_ask_rex[0]);
        $bestbid_rex=$orderbook_bid_rex[0]->Rate;
        $bestask_rex=$orderbook_ask_rex[0]->Rate;
        $bidqty_rex =$orderbook_bid_rex[0]->Quantity;
        $askqty_rex =$orderbook_ask_rex[0]->Quantity;
        echo "Rex : bestbid: $bestbid_rex    bestask: $bestask_rex   $currency_2/$currency_1\n";
        echo "Rex :  bidqty: $bidqty_rex      askqty: $askqty_rex   $currency_1\n";
        $tradable_amount_bid_rex = min($balance_cur_1_rex, $bidqty_rex);
        $tradable_amount_ask_rex = min($balance_cur_2_rex/$bestask_rex, $askqty_rex);
        echo "Rex : tradable amount bid: $tradable_amount_bid_rex  ask: $tradable_amount_ask_rex  $currency_1\n";
        
        echo "\nAnalysis...\n";

        if (($tradable_amount_ask>0) && ($tradable_amount_bid_rex>0)

           ) {
        	echo "---";
                echo "Analyzing Buy on Poloniex, Sell on Bittrex\n";
                $tradable_A = min($tradable_amount_ask, $tradable_amount_bid_rex);
                echo "Tradable: $tradable_A $currency_1\n";
                $gain_A = (1-$fee_rex_taker)*($tradable_A*$bestbid_rex) // sell on Rex
                          -
                          (1+$fee_polo_taker)*($tradable_A*$bestask);     // buy on Poloniex
                $gain_A_in_ref = $gain_A * ($price_2_in_ref+$price_2_in_ref_rex)/2;
                echo "Gain A: $gain_A $currency_2     $gain_A_in_ref $currency_ref\n";
	        $transize_A_in_ref = $tradable_A * ($price_1_in_ref+$price_1_in_ref_rex)/2;
                echo "Transaction size: $transize_A_in_ref $currency_ref ";
                if ($transize_A_in_ref>=$trans_treshold_in_ref) {
			echo "OK\n";
                }
                else   {
			echo "NOT OK\n";
                }
                        
                echo "---\n";
        } else {
             $gain_A = -1;
             $gain_A_in_ref = -1;
             echo "Not possible: Buy on Poloniex, Sell on Bittrex\n";
        }
        
        if (($tradable_amount_ask_rex>0) && ($tradable_amount_bid>0)) {
		echo "---";
                echo "Analyzing Buy on Bittrex, Sell on Poloniex\n";
                $tradable_B = min($tradable_amount_ask_rex, $tradable_amount_bid);
                echo "Tradable: $tradable_B $currency_1\n";
                $gain_B = (1-$fee_polo_taker)*($tradable_B*$bestbid)
                           -
                          (1+$fee_rex_taker)*($tradable_B*$bestask_rex);
                $gain_B_in_ref = $gain_B * ($price_2_in_ref+$price_2_in_ref_rex)/2;
                echo "Gain B: $gain_B $currency_2    $gain_B_in_ref $currency_ref\n";
		$transize_B_in_ref = $tradable_B * ($price_1_in_ref + $price_1_in_ref_rex)/2;
                echo "Transaction size: $transize_B_in_ref $currency_ref ";
                if ($transize_B_in_ref>=$trans_treshold_in_ref) {
                       echo "OK\n";
                }
                else   {
                       echo "NOT OK\n";
                }

                echo "---\n";
        } else {
             $gain_B = -1;
             $gain_B_in_ref = -1;
             echo "Not possible: Buy on Bittrex, Sell on Poloniex\n";
        }

        echo "\nTrading...\n"; 
        if (($gain_A>0) || ($gain_B>0)) {
		if ($gain_A>$gain_B)  {
                     if ($transize_A_in_ref>=$trans_treshold_in_ref) {
                        // buy on poloniex, sell on bittrex
                        $api_polo->buy($curpair_1_2, $bestask, $tradable_A);
                        $api_rex->sellLimit($curpair_1_2_rex, $tradable_A, $bestbid_rex);
                        echo "Order: BUY $tradable_A $currency_1 on Poloniex at $bestask, SELL on Bittrex at $bestbid_rex";
                     }
                  } else {
                     if ($transize_B_in_ref>=$trans_treshold_in_ref) {
                        // buy on bittrex, sell on poloniex
                        $api_rex->buyLimit($curpair_1_2_rex, $tradable_B, $bestask_rex);
                        $api_polo->sell($curpair_1_2, $bestbid, $tradable_B);
                        echo "Order: BUY $tradable_B $currency_1 on Bittrex at $bestask_rex, SELL on Poloniex at $bestbid";
                     }
                }
          
        } else {
		echo "Nothing to do, no winning arbitrage...\n";
        }

        /*
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
			echo "No existing order is appealing to us...\n";
	} else {
			if ($new_portfolio_value_ref_buy>$new_portfolio_value_ref_sell) {
					// we do buy
					echo "We do BUY $tradable_amount_ask $currency_1\n";
					echo "Portfolio value should go to $new_portfolio_value_ref_buy $currency_ref\n";
					$api->buy($curpair_1_2, $bestask, $tradable_amount_ask);
			} else {
					// we do sell
					echo "We do SELL $tradable_amount_sell $currency_1\n";
					echo "Portfolio value should go to $new_portfolio_value_ref_sell $currency_ref\n";
					$api->sell($curpair_1_2, $bestbid, $tradable_amount_sell);
			}	
			
	}
	*/
	echo "Bot iteration over... \n\n";

        $iter=$iter+1;
        sleep(10);
   }
?>
