-- phpMyAdmin SQL Dump
-- version 3.5.2
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Jul 09, 2013 at 04:13 PM
-- Server version: 5.5.25a
-- PHP Version: 5.4.4

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `finance`
--

-- --------------------------------------------------------

--
-- Table structure for table `tickernames`
--

CREATE TABLE IF NOT EXISTS `tickernames` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(4) NOT NULL,
  `description` varchar(255) NOT NULL,
  `type` varchar(16) DEFAULT NULL,
  `url` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=3 ;

--
-- Dumping data for table `tickernames`
--

INSERT INTO `tickernames` (`id`, `name`, `description`, `type`, `url`) VALUES
(1, 'VIX', 'VOLATILITY S&P 500', 'YAHOO', 'http://finance.yahoo.com/q?s=^VIX'),
(2, 'GSPC', 'Standard&Poors 500', 'YAHOO', 'http://finance.yahoo.com/q?s=^gspc');

-- --------------------------------------------------------

--
-- Table structure for table `tickers`
--

CREATE TABLE IF NOT EXISTS `tickers` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `create_dt` datetime NOT NULL,
  `name` varchar(4) NOT NULL,
  `value` double NOT NULL,
  `changepct` double DEFAULT NULL,
  `create_user` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=6 ;

--
-- Dumping data for table `tickers`
--

INSERT INTO `tickers` (`id`, `create_dt`, `name`, `value`, `changepct`, `create_user`) VALUES
(1, '2013-07-09 15:58:59', 'VIX', 14.78, 0, 'php'),
(2, '2013-07-09 15:59:49', 'VIX', 14.52, 0, 'php'),
(3, '2013-07-09 16:08:17', 'VIX', 14.37, 0, 'php'),
(4, '2013-07-09 16:08:21', 'VIX', 14.37, 0, 'php'),
(5, '2013-07-09 16:09:01', 'GSPC', 1645.09, 0, 'php');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
