from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from enum import IntEnum

import gym
import numpy as np
from numpy import random as rd

import matplotlib.pyplot as plt
import matplotlib.cm as plt_cm
import matplotlib.colors as plt_colors
import plotly.graph_objects as go


class OrderType(IntEnum):
    Sell = 0
    Buy = 1

    @property
    def sign(self) -> float:
        return 1. if self == OrderType.Buy else -1.

    @property
    def opposite(self) -> 'OrderType':
        if self == OrderType.Sell:
            return OrderType.Buy
        return OrderType.Sell

class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        # turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.tick_list = config["tick_list"]
        self.time_points = config["time_points"]
        # self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        # self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        # self.turbulence_ary = (
        #     self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        # ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None
        self.time_step = 0
        self.history = []

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        # self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.state_dim = 1 + 3 * stock_dim + self.tech_ary.shape[1]
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):

        print("original actions: ", actions)

        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        # if self.turbulence_bool[self.day] == 0:
        # if True :
        min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd

        print('self.max_stock: ', self.max_stock)
        print('self.min_stock_rate: ', self.min_stock_rate)
        print('actions: ', actions)
        print('min_action: ', min_action)
        print('price: ', price)

        print('np.where(actions < -min_action): ', np.where(actions < -min_action))
        ## TODO: Multiprocess for sell and buy at the same time

        for index in np.where(actions < -min_action)[0]:  # sell_index:
            print('sell: ', index)
            if price[index] > 0:  # Sell only if current asset is > 0
                # sell_num_shares = min(self.stocks[index], -actions[index])
                sell_num_shares = 0.1
                self.stocks[index] -= sell_num_shares
                self.amount += (
                    price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                )
                self.stocks_cool_down[index] = 0

        for index in np.where(actions > min_action)[0]:  # buy_index:
            print('buy: ', index)
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                print('self.amount: ', self.amount)
                print('self.amount // price[index]: ', self.amount // price[index])
                print('actions[index]: ', actions[index])
                # buy_num_shares = min(self.amount // price[index], actions[index])
                buy_num_shares = 0.1
                print('buy_num_shares: ', buy_num_shares)
                self.stocks[index] += buy_num_shares
                self.amount -= (
                    price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                print('self.amount_2: ', self.amount)
                self.stocks_cool_down[index] = 0

        # else:  # sell all when turbulence
        #     self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
        #     self.stocks[:] = 0
        #     self.stocks_cool_down[:] = 0

        print('self.total_asset: ', self.total_asset)
        print('self.stocks: ', self.stocks)
        print('self.stocks * price: ', self.stocks * price)
        print('(self.stocks * price).sum(): ', (self.stocks * price).sum())

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset


        print('total_asset: ', total_asset)
        print('reward: ', reward)

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step

        print('self.gamma_reward: ',  self.gamma_reward)

        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

            print('self.episode_return: ', self.episode_return)

        return state, reward, done, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                # self.turbulence_ary[self.day],
                # self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    def render(self, mode: str='human', **kwargs: Any) -> Any:
        if mode == 'simple_figure':
            return self._render_simple_figure(**kwargs)
        # if mode == 'advanced_figure':
        #     return self._render_advanced_figure(**kwargs)
        # return self.simulator.get_state(**kwargs)
        return None
        

    def _render_simple_figure(
        self, figsize: Tuple[float, float]=(14, 6), return_figure: bool=False
    ) -> Any:
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.tick_list)))

        for j, symbol in enumerate(self.tick_list):
            close_price = self.price_ary[:, j]
            symbol_color = symbol_colors[j]

            ax.plot(self.time_points, close_price, c=symbol_color, marker='.', label=symbol)

            # buy_ticks = []
            # buy_error_ticks = []
            # sell_ticks = []
            # sell_error_ticks = []
            # close_ticks = []

            # for i in range(1, len(self.history)):
            #     tick = self._start_tick + i - 1

            #     order = self.history[i]['orders'].get(symbol, {})
            #     if order and not order['hold']:
            #         if order['order_type'] == OrderType.Buy:
            #             if order['error']:
            #                 buy_error_ticks.append(tick)
            #             else:
            #                 buy_ticks.append(tick)
            #         else:
            #             if order['error']:
            #                 sell_error_ticks.append(tick)
            #             else:
            #                 sell_ticks.append(tick)

            #     closed_orders = self.history[i]['closed_orders'].get(symbol, [])
            #     if len(closed_orders) > 0:
            #         close_ticks.append(tick)

            # tp = np.array(self.time_points)
            # ax.plot(tp[buy_ticks], close_price[buy_ticks], '^', color='green')
            # ax.plot(tp[buy_error_ticks], close_price[buy_error_ticks], '^', color='gray')
            # ax.plot(tp[sell_ticks], close_price[sell_ticks], 'v', color='red')
            # ax.plot(tp[sell_error_ticks], close_price[sell_error_ticks], 'v', color='gray')
            # ax.plot(tp[close_ticks], close_price[close_ticks], '|', color='black')

            ax.tick_params(axis='y', labelcolor=symbol_color)
            # ax.yaxis.tick_left()
            if j < len(self.tick_list) - 1:
                ax = ax.twinx()

        # fig.suptitle(
        #     f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
        #     f"Equity: {self.simulator.equity:.6f} ~ "
        #     f"Margin: {self.simulator.margin:.6f} ~ "
        #     f"Free Margin: {self.simulator.free_margin:.6f} ~ "
        #     f"Margin Level: {self.simulator.margin_level:.6f}"
        # )
        fig.legend()
        # fig.tight_layout()

        if return_figure:
            return fig

        plt.show()


    def _render_advanced_figure(
            self, figsize: Tuple[float, float]=(1400, 600), time_format: str="%Y-%m-%d %H:%m",
            return_figure: bool=False
        ) -> Any:

        fig = go.Figure()

        cmap_colors = np.array(plt_cm.tab10.colors)[[0, 1, 4, 5, 6, 8]]
        cmap = plt_colors.LinearSegmentedColormap.from_list('mtsim', cmap_colors)
        symbol_colors = cmap(np.linspace(0, 1, len(self.trading_symbols)))
        get_color_string = lambda color: "rgba(%s, %s, %s, %s)" % tuple(color)

        extra_info = [
            f"balance: {h['balance']:.6f} {self.simulator.unit}<br>"
            f"equity: {h['equity']:.6f}<br>"
            f"margin: {h['margin']:.6f}<br>"
            f"free margin: {h['free_margin']:.6f}<br>"
            f"margin level: {h['margin_level']:.6f}"
            for h in self.history
        ]
        extra_info = [extra_info[0]] * (self.window_size - 1) + extra_info

        for j, symbol in enumerate(self.trading_symbols):
            close_price = self.prices[symbol][:, 0]
            symbol_color = symbol_colors[j]

            fig.add_trace(
                go.Scatter(
                    x=self.time_points,
                    y=close_price,
                    mode='lines+markers',
                    line_color=get_color_string(symbol_color),
                    opacity=1.0,
                    hovertext=extra_info,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.update_layout(**{
                f'yaxis{j+1}': dict(
                    tickfont=dict(color=get_color_string(symbol_color * [1, 1, 1, 0.8])),
                    overlaying='y' if j > 0 else None,
                    # position=0.035*j
                ),
            })

            trade_ticks = []
            trade_markers = []
            trade_colors = []
            trade_sizes = []
            trade_extra_info = []
            trade_max_volume = max([
                h.get('orders', {}).get(symbol, {}).get('modified_volume') or 0
                for h in self.history
            ])
            close_ticks = []
            close_extra_info = []

            for i in range(1, len(self.history)):
                tick = self._start_tick + i - 1

                order = self.history[i]['orders'].get(symbol)
                if order and not order['hold']:
                    marker = None
                    color = None
                    size = 8 + 22 * (order['modified_volume'] / trade_max_volume)
                    info = (
                        f"order id: {order['order_id'] or ''}<br>"
                        f"hold probability: {order['hold_probability']:.4f}<br>"
                        f"hold: {order['hold']}<br>"
                        f"volume: {order['volume']:.6f}<br>"
                        f"modified volume: {order['modified_volume']:.4f}<br>"
                        f"fee: {order['fee']:.6f}<br>"
                        f"margin: {order['margin']:.6f}<br>"
                        f"error: {order['error']}"
                    )

                    if order['order_type'] == OrderType.Buy:
                        marker = 'triangle-up'
                        color = 'gray' if order['error'] else 'green'
                    else:
                        marker = 'triangle-down'
                        color = 'gray' if order['error'] else 'red'

                    trade_ticks.append(tick)
                    trade_markers.append(marker)
                    trade_colors.append(color)
                    trade_sizes.append(size)
                    trade_extra_info.append(info)

                closed_orders = self.history[i]['closed_orders'].get(symbol, [])
                if len(closed_orders) > 0:
                    info = []
                    for order in closed_orders:
                        info_i = (
                            f"order id: {order['order_id']}<br>"
                            f"order type: {order['order_type'].name}<br>"
                            f"close probability: {order['close_probability']:.4f}<br>"
                            f"margin: {order['margin']:.6f}<br>"
                            f"profit: {order['profit']:.6f}"
                        )
                        info.append(info_i)
                    info = '<br>---------------------------------<br>'.join(info)

                    close_ticks.append(tick)
                    close_extra_info.append(info)

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[trade_ticks],
                    y=close_price[trade_ticks],
                    mode='markers',
                    hovertext=trade_extra_info,
                    marker_symbol=trade_markers,
                    marker_color=trade_colors,
                    marker_size=trade_sizes,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(self.time_points)[close_ticks],
                    y=close_price[close_ticks],
                    mode='markers',
                    hovertext=close_extra_info,
                    marker_symbol='line-ns',
                    marker_color='black',
                    marker_size=7,
                    marker_line_width=1.5,
                    name=symbol,
                    yaxis=f'y{j+1}',
                    showlegend=False,
                    legendgroup=f'g{j+1}',
                ),
            )

        title = (
            f"Balance: {self.simulator.balance:.6f} {self.simulator.unit} ~ "
            f"Equity: {self.simulator.equity:.6f} ~ "
            f"Margin: {self.simulator.margin:.6f} ~ "
            f"Free Margin: {self.simulator.free_margin:.6f} ~ "
            f"Margin Level: {self.simulator.margin_level:.6f}"
        )
        fig.update_layout(
            title=title,
            xaxis_tickformat=time_format,
            width=figsize[0],
            height=figsize[1],
        )

        if return_figure:
            return fig

        fig.show()


    def close(self) -> None:
        plt.close()

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
