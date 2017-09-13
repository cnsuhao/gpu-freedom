<?php
/*
   A simple arbitrage trading bot 
   (c) by 2017 dangermouse (HB9TVM) 
   Source code under GNU Public Licence (GPL)

   Any currency available on Poloniex, Bittrex and C-Cex
   can be configured below.
   
   If you like this bot consider donating 
   gridcoins to this gridcoin address:
   SEtxQ4SePHSP2xSfjy4MATvCWGoFMCcahn
   and send an email to
   dangermaus@users.sourceforge.net
   to notify the transaction.   

   Thank you!
   
*/

  require_once("../lib/poloniex_api.php");
  require_once("../lib/bittrex_api.php");
  require_once("../lib/ccex_api.php");
  require_once("conf/config.inc.php");	

function microtime_float()
{
    list($usec, $sec) = explode(" ", microtime());
    return ((float)$usec + (float)$sec);
}
$time_start = microtime_float();


    
  $iter = 0;
  while ($iter<5) {
    if ((microtime_float()-$time_start)>46) die("timeout 0 reached.");

  
	$date = date('Y-m-d H:i:s');
	echo "$date  iter $iter\n";
	

	// fee structure
	$fee_polo_maker = 0.0015;
	$fee_polo_taker = 0.0025;
        
    $fee_rex_maker  = 0.0025;
    $fee_rex_taker  = 0.0025;

	$fee_cex_maker  = 0.0015;
    $fee_cex_taker  = 0.0020;

	
    $trans_treshold_in_ref = 3.0; // transactions have to be at least
                                    // this amount in $currency_ref
	
    // currency to be arbitraged
    $currency_1 = "GRC";
    $max_tradable_1 = 750; // maximum amount tradable in currency 1

    $currency_2 = "BTC"; // currency 2
    $max_tradable_2 = 0.0075; // maximum amount tradable in currency 2

    $currency_ref = "USDT"; // used to get an idea of portfolio value
        

    // currency pairs
    $curpair_1_2 = $currency_2 . "_" . $currency_1;
    $curpair_2_ref = $currency_ref . "_" . $currency_2;
    $curpair_1_2_rex = $currency_2 . "-" . $currency_1;
    $curpair_2_ref_rex = $currency_ref . "-" . $currency_2;
    if ($currecy_ref="USDT") $currency_ref_cex="USD"; else $currency_ref_cex=$currency_ref;
    $curpair_1_2_cex = strtolower($currency_1) . "-" . strtolower($currency_2);
    $curpair_2_ref_cex = strtolower($currency_ref_cex) . "-" . strtolower($currency_2);   

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
		
	echo "API C-CEX key: ";
        echo $ccex_api_key;
        echo "\n";
        echo "API C-CEX secret: ";
	echo $ccex_api_secret;
        echo "\n";
    */
        
	$api_polo = new poloniex($poloniex_api_key, $poloniex_api_secret);
        $api_rex  = new bittrex_api($bittrex_api_key, $bittrex_api_secret);	
	$api_cex  = new ccex_api($ccex_api_key, $ccex_api_secret);	
 
      
	// 0. cancel existing orders lying around from previous bot calls 
	// (they lay around for example if the order could be only partially fullfilled)
	// get_open_orders($pair), retrieves ordernumber
	// iterate over and do cancel_order($pair, $order_number)
    //echo "Retrieving open orders on Poloniex...\n";
    $openorders = $api_polo->get_open_orders($curpair_1_2);
	
    //echo "* openorders poloniex result: \n";
    //print_r($openorders);
    //echo "*\n";
    echo "Open orders on Poloniex ";
    echo count($openorders[0]);
    echo "\n";
	
    if (count($openorders[0])>0) { //hack due to API inconsistency
            echo "WARNING: there are open orders on Poloniex\n";

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
	
    //echo "Retrieving open orders on bittrex\n";
    $openorders_rex = $api_rex->getOpenOrders($curpair_1_2_rex);
    //echo "* openorders bittrex result: \n";
    //print_r($openorders_rex);
    echo "Count open orders bittrex: "; 
    
    echo count($openorders_rex);
    echo "\n";
	if (count($openorders_rex)>0) {
	    echo "WARNING: there are open orders on Rex\n";
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
	}
    
    
    //echo "Retrieving open orders on c-cex";
    $openorders_cex = $api_cex->getOpenOrders($curpair_1_2_rex);
    echo "Openorders ccex result: ";
    //print_r($openorders_cex);
    
    //echo "*\n"; 
    echo count($openorders_cex);
    echo "\n";
    
	if (count($openorders_cex)>0) {
	    echo "WARNING: there are open orders on Cex\n";
        $i=0;
        while ($i<count($openorders_cex)) {
                $orderid_cex = $openorders_cex[$i]->OrderUuid;
                $quantity_cex = $openorders_cex[$i]->Quantity;
                if ($quantity_cex<=$max_tradable_1) {
                     echo "Cancelling cex order $orderid_cex ($quantity_cex $currency_1) \n";
                     $api_cex->cancel($orderid_cex);
                } else {
                     echo "Not cancelling cex order $orderid_cex ($quantity_cex $currency_1 because not done by bot \n";
                }
                $i=$i+1;
      }
	}
    if ((microtime_float()-$time_start)>49) die("timeout 1 reached.");
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
	
    //$myres_markets = $api_cex->getMarkets();
    //print_r($myres_markets);	
    $myres_curpair_1_2_cex = $api_cex->getTicker($curpair_1_2_cex);
    //print_r($myres_curpair_1_2_cex);
    
    $myres_curpair_2_ref_cex = $api_cex->getTicker($curpair_2_ref_cex);
    //print_r($myres_curpair_2_ref_cex);        
    
    $price_1_in_2_cex = $myres_curpair_1_2_cex->lastprice;
    $price_2_in_ref_cex = 1/$myres_curpair_2_ref_cex->lastprice;
    $price_1_in_ref_cex = $price_1_in_2_cex * $price_2_in_ref_cex;

    echo "Ccex: $price_1_in_ref_cex $currency_1/$currency_ref_cex $price_2_in_ref_cex  $currency_ref_cex/$currency_2 $price_1_in_2_cex $currency_2/$currency_1\n";
    echo "\n";
	
	if ((microtime_float()-$time_start)>52) die("timeout 2 reached.");
        
        
	// 2. retrieve our current balance in currency 1 and 2
	//    and calculate current portfolio value in reference currency
	//sleep(1);
    $balances = $api_polo->get_balances();
	//print_r($balances);
    //echo $balances["GRC"]["available"];
    //echo "\n";
    //echo $balances["BTC"]["available"];
    //echo "\n";

	$balance_polo_tot_1 = $balances[$currency_1]["available"];
	$balance_polo_tot_2 = $balances[$currency_2]["available"];
    $balance_cur_1 = min($balance_polo_tot_1,$max_tradable_1);
	$balance_cur_2 = min($balance_polo_tot_2,$max_tradable_2);
	$cur_portfolio_value_ref = ($balance_cur_1 * $price_1_in_ref) + ($balance_cur_2 * $price_2_in_ref);	
	$cur_portfolio_value_polo_ref_tot = ($balance_polo_tot_1 * $price_1_in_ref) + ($balance_polo_tot_2 * $price_2_in_ref);	
	$cur_portfolio_value_polo_1_tot = $balance_polo_tot_1 + ($balance_polo_tot_2/$price_1_in_2);
	
	echo "Polo: $balance_polo_tot_1 $currency_1 + $balance_polo_tot_2 $currency_2 -> $cur_portfolio_value_polo_ref_tot $currency_ref (=$cur_portfolio_value_polo_1_tot $currency_1)\n";
	
	$balances_rex = $api_rex->getBalances();
	//echo "Balances Rex.\";
	//print_r($balances_rex);
	//echo "* \n";
	//echo "Count: ";
	//echo count($balances_rex);
	//echo "\n";
    // only one call to getBalances instead of two calls to getBalance!	
	$balances_rex_tot_1 = -1;
	$balances_rex_tot_2 = -1;
	$i=0;
	
	while ($i<count($balances_rex)) {
                $currency_rex = $balances_rex[$i]->Currency;
                $available_rex = $balances_rex[$i]->Available;
                if ($currency_rex==$currency_1) {
                     $balances_rex_tot_1 = $available_rex;
                } else 
				if ($currency_rex==$currency_2) {
                     $balances_rex_tot_2 = $available_rex;
                }	
                $i=$i+1;
    }
	if (($balances_rex_tot_1==-1) || ($balances_rex_tot_2==-1)) die("Internal ERROR: could not retrieve balances on Rex!\n");
	
	
    //$balances_rex_tot_1 = $api_rex->getBalance($currency_1)->Available;
    //$balances_rex_tot_2 = $api_rex->getBalance($currency_2)->Available;
    //print_r($balances_rex_1);
    $balance_cur_1_rex = min($balances_rex_tot_1, $max_tradable_1);
    $balance_cur_2_rex = min($balances_rex_tot_2, $max_tradable_2);
    $cur_portfolio_value_ref_rex = ($balance_cur_1_rex * $price_1_in_ref_rex) + ($balance_cur_2_rex * $price_2_in_ref_rex);
    $cur_portfolio_value_rex_ref_tot = ($balances_rex_tot_1 * $price_1_in_ref_rex) + ($balances_rex_tot_2 * $price_2_in_ref_rex);	
	$cur_portfolio_value_rex_1_tot = $balances_rex_tot_1 + ($balances_rex_tot_2/$price_1_in_2_rex);

	echo "Rex : $balances_rex_tot_1 $currency_1 + $balances_rex_tot_2 $currency_2 -> $cur_portfolio_value_rex_ref_tot $currency_ref (=$cur_portfolio_value_rex_1_tot $currency_1)\n";	
    
	
    $balances_cex = $api_cex->getBalances();
	//echo "Balances cex.\n";
	//print_r($balances_cex);
	//echo "* \n";
	//echo "Count: ";
	//echo count($balances_cex);
	//echo "\n";
    // only one call to getBalances instead of two calls to getBalance!	
	$balances_cex_tot_1 = -1;
	$balances_cex_tot_2 = -1;
	$i=0;
	
	while ($i<count($balances_cex)) {
                $currency_cex = $balances_cex[$i]->Currency;
                $available_cex = $balances_cex[$i]->Available;
                if ($currency_cex==$currency_1) {
                     $balances_cex_tot_1 = $available_cex;
                } else 
				if ($currency_cex==$currency_2) {
                     $balances_cex_tot_2 = $available_cex;
                }	
                $i=$i+1;
    }
	if (($balances_cex_tot_1==-1) || ($balances_cex_tot_2==-1)) die("Internal ERROR: could not retrieve balances on Cex!\n");
	
	
    //$balances_cex_tot_1 = $api_cex->getBalance($currency_1)->Available;
    //sleep(2);  // the two calls need to be spaced by 1 second TODO: implement getBalances
    //$balances_cex_tot_2 = $api_cex->getBalance($currency_2)->Available;
    //echo "*\n";
    //print_r($balances_cex_1);
    //echo "\*\n";
    //print_r($balances_cex_2);
    //echo "\n*\n";    
    
    $balance_cur_1_cex = min($balances_cex_tot_1, $max_tradable_1);
    $balance_cur_2_cex = min($balances_cex_tot_2, $max_tradable_2);
    $cur_portfolio_value_ref_cex = ($balance_cur_1_cex * $price_1_in_ref_cex) + ($balance_cur_2_cex * $price_2_in_ref_cex);
    $cur_portfolio_value_cex_ref_tot = ($balances_cex_tot_1 * $price_1_in_ref_cex) + ($balances_cex_tot_2 * $price_2_in_ref_cex);	
	$cur_portfolio_value_cex_1_tot = $balances_cex_tot_1 + ($balances_cex_tot_2/$price_1_in_2_cex);

	echo "Ccex : $balances_cex_tot_1 $currency_1 + $balances_cex_tot_2 $currency_2 -> $cur_portfolio_value_cex_ref_tot $currency_ref (=$cur_portfolio_value_cex_1_tot $currency_1)\n";	
	
	
	$balances_tot_1 = $balance_polo_tot_1 + $balances_rex_tot_1 + $balances_cex_tot_1;
	$balances_tot_2 = $balance_polo_tot_2 + $balances_rex_tot_2 + $balances_cex_tot_2;
	$tot_portfolio_value_ref = ($balances_tot_1 * $price_1_in_ref) + ($balances_tot_2 * $price_2_in_ref);	
	$tot_portfolio_value_1  = $balances_tot_1 + ($balances_tot_2/$price_1_in_2);
    $tot_portfolio_value_2  = $balances_tot_1*$price_1_in_2 + $balances_tot_2;
    
	echo "Total: $balances_tot_1 $currency_1 + $balances_tot_2 $currency_2 ->\n"; 	
	echo "Total: $tot_portfolio_value_ref $currency_ref = $tot_portfolio_value_1 $currency_1 = $tot_portfolio_value_2 $currency_2\n";
	echo "\n";
     if ((microtime_float()-$time_start)>55) die("timeout 3 reached.");
    
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
	if ((microtime_float()-$time_start)>57) die("timeout 4.1 reached.");
    
    
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
    echo "------------------------\n";    
	
	if ((microtime_float()-$time_start)>59) die("timeout 4.2 reached.");
    
    $orderbook_bid_cex = $api_cex->getOrderBook($curpair_1_2_cex, "buy" /* or buy or sell*/, 1 /*market depth*/)->buy;
    //print_r($orderbook_bid_cex[0]);
    $orderbook_ask_cex = $api_cex->getOrderBook($curpair_1_2_cex, "sell", 1)->sell;
    //print_r($orderbook_ask_cex[0]);
    $bestbid_cex=$orderbook_bid_cex[0]->Rate;
    $bestask_cex=$orderbook_ask_cex[0]->Rate;
    $bidqty_cex =$orderbook_bid_cex[0]->Quantity;
    $askqty_cex =$orderbook_ask_cex[0]->Quantity;
    echo "Ccex: bestbid: $bestbid_cex    bestask: $bestask_cex   $currency_2/$currency_1\n";
    echo "Ccex:  bidqty: $bidqty_cex      askqty: $askqty_cex   $currency_1\n";
    $tradable_amount_bid_cex = min($balance_cur_1_cex, $bidqty_cex);
    $tradable_amount_ask_cex = min($balance_cur_2_cex/$bestask_cex, $askqty_cex);
    echo "Ccex: tradable amount bid: $tradable_amount_bid_cex  ask: $tradable_amount_ask_cex  $currency_1\n";
     
    echo "\nAnalysis...\n";

     if (($tradable_amount_ask>0) && ($tradable_amount_bid_rex>0)) {
        	echo "---";
                echo "A: Analyzing Buy on Poloniex, Sell on Bittrex\n";
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
						$gain_A = -1;
                        $gain_A_in_ref = -1;
                }
                        
                echo "---\n";
        } else {
             $gain_A = -1;
             $gain_A_in_ref = -1;
             echo "Not possible: Buy on Poloniex, Sell on Bittrex: no money.\n";
        }
        
        if (($tradable_amount_ask_rex>0) && ($tradable_amount_bid>0)) {
		echo "---";
                echo "B: Analyzing Buy on Bittrex, Sell on Poloniex\n";
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
					    $gain_B = -1;
                        $gain_B_in_ref = -1;
                }

                echo "---\n";
        } else {
             $gain_B = -1;
             $gain_B_in_ref = -1;
             echo "Not possible: Buy on Bittrex, Sell on Poloniex: no money.\n";
        }
		
		if (($tradable_amount_ask>0) && ($tradable_amount_bid_cex>0)) {
        	echo "---";
                echo "C: Analyzing Buy on Poloniex, Sell on Cex\n";
                $tradable_C = min($tradable_amount_ask, $tradable_amount_bid_cex);
                echo "Tradable: $tradable_C $currency_1\n";
                $gain_C = (1-$fee_cex_taker)*($tradable_C*$bestbid_cex) // sell on cex
                          -
                          (1+$fee_polo_taker)*($tradable_C*$bestask);     // buy on Poloniex
                $gain_C_in_ref = $gain_C * ($price_2_in_ref+$price_2_in_ref_cex)/2;
                echo "Gain C: $gain_C $currency_2     $gain_C_in_ref $currency_ref_cex\n";
	            $transize_C_in_ref = $tradable_C * ($price_1_in_ref+$price_1_in_ref_cex)/2;
                echo "Transaction size: $transize_C_in_ref $currency_ref ";
                if ($transize_C_in_ref>=$trans_treshold_in_ref) {
			        echo "OK\n";
                }
                else   {
					echo "NOT OK\n";
					$gain_C = -1;
                    $gain_C_in_ref = -1;
                }
                        
                echo "---\n";
        } else {
             $gain_C = -1;
             $gain_C_in_ref = -1;
             echo "Not possible: Buy on Poloniex, Sell on Cex: no money.\n";
        }
		
		if (($tradable_amount_ask_cex>0) && ($tradable_amount_bid>0)) {
		echo "---";
                echo "D: Analyzing Buy on Cex, Sell on Poloniex\n";
                $tradable_D = min($tradable_amount_ask_cex, $tradable_amount_bid);
                echo "Tradable: $tradable_D $currency_1\n";
                $gain_D = (1-$fee_polo_taker)*($tradable_D*$bestbid)
                           -
                          (1+$fee_cex_taker)*($tradable_D*$bestask_cex);
                $gain_D_in_ref = $gain_D * ($price_2_in_ref+$price_2_in_ref_cex)/2;
                echo "Gain D: $gain_D $currency_2    $gain_D_in_ref $currency_ref_cex\n";
		        $transize_D_in_ref = $tradable_D * ($price_1_in_ref + $price_1_in_ref_cex)/2;
                echo "Transaction size: $transize_D_in_ref $currency_ref ";
                if ($transize_D_in_ref>=$trans_treshold_in_ref) {
                       echo "OK\n";
                }
                else   {
                       echo "NOT OK\n";
					    $gain_D = -1;
                        $gain_D_in_ref = -1;
                }

                echo "---\n";
        } else {
             $gain_D = -1;
             $gain_D_in_ref = -1;
             echo "Not possible: Buy on Cex, Sell on Poloniex: no money.\n";
        }
		
		
		if (($tradable_amount_ask_rex>0) && ($tradable_amount_bid_cex>0)) {
        	    echo "---";
                echo "E: Analyzing Buy on Bittrex, Sell on Cex\n";
                $tradable_E = min($tradable_amount_ask_rex, $tradable_amount_bid_cex);
                echo "Tradable: $tradable_E $currency_1\n";
                $gain_E = (1-$fee_cex_taker)*($tradable_E*$bestbid_cex) // sell on cex
                          -
                          (1+$fee_rex_taker)*($tradable_E*$bestask_rex);     // buy on bittrex
                $gain_E_in_ref = $gain_E * ($price_2_in_ref_rex+$price_2_in_ref_cex)/2;
                echo "Gain E: $gain_E $currency_2     $gain_E_in_ref $currency_ref_cex\n";
	            $transize_E_in_ref = $tradable_E * ($price_1_in_ref_rex+$price_1_in_ref_cex)/2;
                echo "Transaction size: $transize_E_in_ref $currency_ref ";
                if ($transize_E_in_ref>=$trans_treshold_in_ref) {
			        echo "OK\n";
                }
                else   {
					echo "NOT OK\n";
					$gain_E = -1;
                    $gain_E_in_ref = -1;
                }
                        
                echo "---\n";
        } else {
             $gain_E = -1;
             $gain_E_in_ref = -1;
             echo "Not possible: Buy on Bittrex, Sell on Cex: no money.\n";
        }
		
		if (($tradable_amount_ask_cex>0) && ($tradable_amount_bid_rex>0)) {
		echo "---";
                echo "F: Analyzing Buy on Cex, Sell on Bittrex\n";
                $tradable_F = min($tradable_amount_ask_cex, $tradable_amount_bid_rex);
                echo "Tradable: $tradable_F $currency_1\n";
                $gain_F = (1-$fee_rex_taker)*($tradable_F*$bestbid_rex)
                           -
                          (1+$fee_cex_taker)*($tradable_F*$bestask_cex);
                $gain_F_in_ref = $gain_F * ($price_2_in_ref_rex+$price_2_in_ref_cex)/2;
                echo "Gain F: $gain_F $currency_2    $gain_F_in_ref $currency_ref_cex\n";
		        $transize_F_in_ref = $tradable_F * ($price_1_in_ref_rex + $price_1_in_ref_cex)/2;
                echo "Transaction size: $transize_F_in_ref $currency_ref ";
                if ($transize_F_in_ref>=$trans_treshold_in_ref) {
                       echo "OK\n";
                }
                else   {
                       echo "NOT OK\n";
					    $gain_F = -1;
                        $gain_F_in_ref = -1;
                }

                echo "---\n";
        } else {
             $gain_F = -1;
             $gain_F_in_ref = -1;
             echo "Not possible: Buy on Cex, Sell on Bittrex: no money.\n";
        }

        		
		$datetrading = date('Y-m-d H:i:s');
		// do we trade?
		if (($gain_A<=0) && ($gain_B<=0) && ($gain_C<=0) && ($gain_D<=0) && ($gain_E<=0) && ($gain_F<=0)) {
				echo "$datetrading: Nothing to do: poloniex, bittrex and cex are already arbitraged...\n";
		} else {

                 
			     // identify which is the best possible trade
				 $trade_A = 0;
				 $trade_B = 0;
				 $trade_C = 0;
				 $trade_D = 0;
				 $trade_E = 0;
				 $trade_F = 0;
				 if (($gain_A>$gain_B) && ($gain_A>$gain_C) && ($gain_A>$gain_D) && ($gain_A>$gain_E) && ($gain_A>$gain_F)) {
				   $trade_A = 1;
				 }
				 else
				 if (($gain_B>$gain_A) && ($gain_B>$gain_C) && ($gain_B>$gain_D) && ($gain_B>$gain_E) && ($gain_B>$gain_F)) {
				   $trade_B = 1;
				 }
				 else
				 if (($gain_C>$gain_A) && ($gain_C>$gain_B) && ($gain_C>$gain_D) && ($gain_C>$gain_E) && ($gain_C>$gain_F)) {
				   $trade_C = 1;
				 }
				 else
				 if (($gain_D>$gain_A) && ($gain_D>$gain_B) && ($gain_D>$gain_C) && ($gain_D>$gain_E) && ($gain_D>$gain_F)) {
				   $trade_D = 1;
				 }
				 else
				 if (($gain_E>$gain_A) && ($gain_E>$gain_B) && ($gain_E>$gain_C) && ($gain_E>$gain_D) && ($gain_E>$gain_F)) {
				   $trade_E = 1;
				 }
				 else
				 if (($gain_F>$gain_A) && ($gain_F>$gain_B) && ($gain_F>$gain_C) && ($gain_F>$gain_D) && ($gain_F>$gain_E)) {
				   $trade_F = 1;
				 }
		
				echo "\nTrading...\n"; 
                               if ($trade_A==1) {
					$api_polo->buy($curpair_1_2, $bestask, $tradable_A);
					$api_rex->sellLimit($curpair_1_2_rex, $tradable_A, $bestbid_rex);
					echo "$datetrading Order: BUY $tradable_A $currency_1 on Poloniex at $bestask, SELL on Bittrex at $bestbid_rex Gain_A: $gain_A_in_ref $currency_ref\n";
				} else
		        if ($trade_B==1) {
					$api_rex->buyLimit($curpair_1_2_rex, $tradable_B, $bestask_rex);
					$api_polo->sell($curpair_1_2, $bestbid, $tradable_B);
					echo "$datetrading Order: BUY $tradable_B $currency_1 on Bittrex at $bestask_rex, SELL on Poloniex at $bestbid Gain_B: $gain_B_in_ref $currency_ref\n";
				} else
                if ($trade_C==1) {
					$api_polo->buy($curpair_1_2, $bestask, $tradable_C);
					$api_cex->sellLimit($curpair_1_2_cex, $tradable_C, $bestbid_cex);
					echo "$datetrading Order: BUY $tradable_C $currency_1 on Poloniex at $bestask, SELL on Cex at $bestbid_cex Gain_C: $gain_C_in_ref $currency_ref\n";
				} else
                if ($trade_D==1) {
					$api_cex->buyLimit($curpair_1_2_cex, $tradable_D, $bestask_cex);
					$api_polo->sell($curpair_1_2, $bestbid, $tradable_D);
					echo "$datetrading Order: BUY $tradable_D $currency_1 on Cex at $bestask_cex, SELL on Poloniex at $bestbid Gain_D: $gain_D_in_ref $currency_ref\n";
				} else
				if ($trade_E==1) {
					$api_rex->buyLimit($curpair_1_2_rex, $tradable_E, $bestask_rex);
					$api_cex->sellLimit($curpair_1_2_cex, $tradable_E, $bestbid_cex);
					echo "$datetrading Order: BUY $tradable_E $currency_1 on Rex at $bestask_rex, SELL on Cex at $bestbid_cex Gain_E: $gain_E_in_ref $currency_ref\n";
				} else
			    if ($trade_F==1) {
					$api_cex->buyLimit($curpair_1_2_cex, $tradable_F, $bestask_cex);
					$api_rex->sellLimit($curpair_1_2_rex, $tradable_F, $bestbid_rex);
					echo "$datetrading Order: BUY $tradable_F $currency_1 on Cex at $bestask_cex, SELL on Bittrex at $bestbid_rex Gain_F: $gain_F_in_ref $currency_ref\n";
				} 
                
		   
		
                } // end of trading section
      
		echo "Bot iteration $iter over... \n\n";
        
        $iter=$iter+1;
        //sleep(10);
   } // end of iteration section

?>
