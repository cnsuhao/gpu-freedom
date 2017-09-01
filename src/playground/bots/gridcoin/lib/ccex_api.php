<?php
/**
 * PHP API for c-cex.com exchange
 *
 * @author dangermouse 
 * derived from
 * @license MIT License - https://github.com/Remdev/PHP-ccex-api
 */
class ccex_api {
    
    private $baseUrl;
	private $apiUrl;
	private $apiPubUrl;
	private $apiKey;
	private $apiSecret;

	public function __construct ($apiKey, $apiSecret)
	{
		$this->apiKey    = $apiKey;
		$this->apiSecret = $apiSecret;
		$this->baseUrl   = "https://c-cex.com/t/";
		$this->apiUrl   = "api.html?a=";
		$this->apiPubUrl = "api_pub.html?a=";
	}

	/**
	 * Invoke API
	 * @param string $method API method to call
	 * @param array $params parameters
	 * @param bool $apiKey  use apikey or not
	 * @return object
	 */
	private function call ($method, $params = array(), $apiKey = false, $ispub = false, $isdirect = false)
	{
		if ($isdirect==true) {
			$uri  = $this->baseUrl.$method; 
		} else {
					if ($ispub==true) 
						$uri  = $this->baseUrl . $this->apiPubUrl . $method;
					else
						$uri  = $this->baseUrl . $this->apiUrl . $method;
		}
		
		
		if ($apiKey == true) {
			$params['apikey'] = $this->apiKey;
			$params['nonce']  = time();
		}

		if ((!empty($params)) && (!$isdirect)) {
			$uri .= '&'.http_build_query($params);
		}

		$sign = hash_hmac ('sha512', $uri, $this->apiSecret);
                echo "\n".$uri."\n";
		$ch = curl_init ($uri);
		curl_setopt ($ch, CURLOPT_HTTPHEADER, array('apisign: '.$sign));
		curl_setopt ($ch, CURLOPT_RETURNTRANSFER, true);
		$result = curl_exec($ch);

                $answer = json_decode($result);

                if ((!$isdirect) && ($answer->success == false)) {
			throw new \Exception ($answer->message);
		}

                if ($isdirect)
                     return $answer;
                else
		     return $answer->result;

	}
	
	   
    public function getTicker($pair){
        $params=array();
        $json = $this->call($pair.'.json', $params, false, false, true);
        return $json->ticker;
    }
    
    
    public function getMarkets(){
       $params=array();
       $json = $this->call('pairs.json',$params, false, false, true); 
       return $json;
    }
    
    /**
	 * Get the orderbook for a given market
	 * @param string $market  literal for the market (ex: BTC-LTC)
	 * @param string $type	  "buy", "sell" or "both" to identify the type of orderbook to return
	 * @param integer $depth  how deep of an order book to retrieve. Max is 50.
	 * @return array
	 */
	public function getOrderBook ($market, $type, $depth = 20)
	{
		$params = array (
			'market' => $market,
			'type'   => $type,
			'depth'  => $depth
		);
		return $this->call ('getorderbook', $params, $ispub=true);
	}
    
    /**
	 * Place a limit buy order in a specific market. 
	 * Make sure you have the proper permissions set on your API keys for this call to work
	 * @param string $market  literal for the market (ex: BTC-LTC)
	 * @param float $quantity the amount to purchase
	 * @param float $rate     the rate at which to place the order
	 * @return array
	 */
	public function buyLimit ($market, $quantity, $rate)
	{
		$params = array (
			'market'   => $market,
			'quantity' => $quantity,
			'rate'     => $rate
		);
		return $this->call ('buylimit', $params, true);
	}
	

	/**
	 * Place a limit sell order in a specific market. 
	 * Make sure you have the proper permissions set on your API keys for this call to work
	 * @param string $market  literal for the market (ex: BTC-LTC)
	 * @param float $quantity the amount to sell
	 * @param float $rate     the rate at which to place the order
	 * @return array
	 */
	public function sellLimit ($market, $quantity, $rate)
	{
		$params = array (
			'market'   => $market,
			'quantity' => $quantity,
			'rate'     => $rate
		);
		return $this->call ('selllimit', $params, true);
	}

	/**
	 * Cancel a buy or sell order 
	 * @param string $uuid id of sell or buy order
	 * @return array
	 */
	public function cancel ($uuid)
	{
		$params = array ('uuid' => $uuid);
		return $this->call ('cancel', $params, true);
	}


	/**
	 * Retrieve all balances from your account
	 * @return array
	 */
	public function getBalances ()
	{
		return $this->call ('getbalances', array(), true);
	}

	/**
	 * Retrieve the balance from your account for a specific currency
	 * @param string $currency literal for the currency (ex: LTC)
	 * @return array
	 */
	public function getBalance ($currency)
	{
		$params = array ('currency' => $currency);
		return $this->call ('getbalance', $params, true);
	}

	
	/**
	 * Retrieve a single order by uuid
	 * @param string $uuid 	the uuid of the buy or sell order
	 * @return array
	 */
	public function getOrder ($uuid)
	{
		$params = array ('uuid' => $uuid);
		return $this->call ('getorder', $params, true);
	}

	/**
	 * Retrieve your order history
	 * @param string $market  (optional) a string literal for the market (ie. BTC-LTC). If ommited, will return for all markets
	 * @param integer $count  (optional) the number of records to return
	 * @return array
	 */
	public function getOrderHistory ($market = null, $count = null)
	{
		$params = array ();

		if ($market) {
			$params['market'] = $market;
		}

		if ($count) {
			$params['count'] = $count;
		}

		return $this->call ('getorderhistory', $params, true);
	}

	/**
	 * Get all orders that you currently have opened. A specific market can be requested
	 * @param string $market  literal for the market (ex: BTC-LTC)
	 * @return array
	 */
	public function getOpenOrders ($market = null)
	{
		$params = array ('market' => $market);
		return $this->call ('getopenorders', $params, true);
	}
	
}
